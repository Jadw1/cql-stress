use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use scylla::client::session::Session;
use scylla::cluster::{ClusterState, NodeRef};
use scylla::policies::load_balancing::{FallbackPlan, LoadBalancingPolicy, RoutingInfo};
use scylla::routing::Shard;
use uuid::Uuid;

/// A load balancing policy that routes requests for a specific table
/// to the Raft leader of the corresponding tablet.
///
/// For all other tables/queries, it delegates to the wrapped default policy.
///
/// This is intended for testing ScyllaDB's strong consistency feature,
/// where each tablet has its own Raft group and requests should be directed
/// to the Raft leader of the tablet that owns the requested token.
///
/// The mapping is static and set once during initialization. It assumes
/// that tablet topology won't change (no splits/merges/migrations) and
/// that the Raft leader won't change during the test.
#[derive(Debug)]
pub struct RaftLeaderPolicy {
    target_keyspace: String,
    target_table: String,
    /// Mapping from last_token -> (leader host_id, shard).
    /// Populated after session creation via `initialize()`.
    token_to_leader: OnceLock<BTreeMap<i64, (Uuid, Shard)>>,
    /// Fallback policy for non-target-table queries or when not yet initialized.
    default_policy: Arc<dyn LoadBalancingPolicy>,
}

impl RaftLeaderPolicy {
    pub fn new(
        target_keyspace: String,
        target_table: String,
        default_policy: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        Self {
            target_keyspace,
            target_table,
            token_to_leader: OnceLock::new(),
            default_policy,
        }
    }

    /// Initialize the policy by querying system.tablets and the ScyllaDB REST API.
    ///
    /// This must be called after the session is built and the schema is created.
    /// After initialization, `pick()` will route requests for the target table
    /// to the Raft leader of the corresponding tablet.
    pub async fn initialize(
        &self,
        session: &Session,
        api_node: &str,
        api_port: u16,
    ) -> Result<()> {
        println!(
            "Initializing RaftLeaderPolicy for {}.{}...",
            self.target_keyspace, self.target_table
        );

        // Step 1: Get table_id from system_schema.tables
        let table_id =
            get_table_id(session, &self.target_keyspace, &self.target_table).await?;
        println!("  Table ID: {}", table_id);

        // Step 2: Get token -> (raft_group_id, replicas) mapping from system.tablets
        let tablet_info = get_tablet_info(session, table_id).await?;
        println!("  Found {} tablets", tablet_info.len());

        // Step 3: For each unique raft_group_id, get the leader host_id via REST API
        let http_client = reqwest::Client::new();
        let unique_groups: std::collections::HashSet<Uuid> =
            tablet_info.iter().map(|t| t.raft_group_id).collect();
        println!(
            "  Querying raft leaders for {} groups...",
            unique_groups.len()
        );

        let mut group_to_leader: std::collections::HashMap<Uuid, Uuid> =
            std::collections::HashMap::new();

        for group_id in &unique_groups {
            let leader_host_id =
                get_raft_leader(&http_client, api_node, api_port, *group_id)
                    .await
                    .with_context(|| {
                        format!("Failed to get raft leader for group {}", group_id)
                    })?;
            println!("    Group {} -> Leader {}", group_id, leader_host_id);
            group_to_leader.insert(*group_id, leader_host_id);
        }

        // Build the final token -> (leader host_id, shard) mapping.
        // For each tablet, find the replica on the leader host to get the shard.
        let mut token_to_leader = BTreeMap::new();
        for tablet in &tablet_info {
            if let Some(&leader_host_id) = group_to_leader.get(&tablet.raft_group_id) {
                let shard = tablet
                    .replicas
                    .iter()
                    .find(|(host_id, _)| *host_id == leader_host_id)
                    .map(|(_, shard)| *shard as Shard)
                    .with_context(|| {
                        format!(
                            "Leader {} not found in replicas for tablet with last_token {}",
                            leader_host_id, tablet.last_token
                        )
                    })?;
                println!(
                    "    Token {} -> Host {} Shard {}",
                    tablet.last_token, leader_host_id, shard
                );
                token_to_leader.insert(tablet.last_token, (leader_host_id, shard));
            }
        }

        println!(
            "  Raft leader mapping initialized: {} tablet entries",
            token_to_leader.len()
        );

        self.token_to_leader
            .set(token_to_leader)
            .map_err(|_| anyhow::anyhow!("RaftLeaderPolicy already initialized"))?;

        Ok(())
    }

    /// Find the leader host_id and shard for a given token.
    ///
    /// Uses the BTreeMap to find the tablet whose range contains the token.
    /// The tablet ranges are defined by `last_token` boundaries:
    /// a token T belongs to the tablet with the smallest `last_token >= T`.
    /// If T is beyond the largest `last_token`, it wraps around to the first tablet.
    fn find_leader(&self, token_value: i64) -> Option<(Uuid, Shard)> {
        let map = self.token_to_leader.get()?;
        if map.is_empty() {
            return None;
        }
        // Find the first tablet whose last_token >= the given token
        map.range(token_value..)
            .next()
            .map(|(_, target)| *target)
            // Wrap around: if token > max last_token, use the first tablet
            .or_else(|| map.values().next().copied())
    }

    /// Check if the request targets the table we're managing.
    fn is_target_table(&self, info: &RoutingInfo) -> bool {
        if let Some(table) = &info.table {
            table.ks_name() == self.target_keyspace
                && table.table_name() == self.target_table
        } else {
            false
        }
    }

    /// Find a node in the cluster by its host_id, paired with a specific shard.
    fn find_node_by_host_id<'a>(
        host_id: Uuid,
        shard: Shard,
        cluster: &'a ClusterState,
    ) -> Option<(NodeRef<'a>, Option<Shard>)> {
        cluster
            .get_nodes_info()
            .iter()
            .find(|node| node.host_id == host_id)
            .map(|node| (node, Some(shard)))
    }
}

impl LoadBalancingPolicy for RaftLeaderPolicy {
    fn pick<'a>(
        &'a self,
        request: &'a RoutingInfo,
        cluster: &'a ClusterState,
    ) -> Option<(NodeRef<'a>, Option<Shard>)> {
        // For the target table, try to route to the raft leader + shard
        if self.is_target_table(request) {
            if let Some(token) = request.token {
                if let Some((leader_id, shard)) = self.find_leader(token.value()) {
                    if let Some(result) = Self::find_node_by_host_id(leader_id, shard, cluster) {
                        return Some(result);
                    }
                }
            }
        }

        // Fallback to default policy for non-target tables or missing info
        self.default_policy.pick(request, cluster)
    }

    fn fallback<'a>(
        &'a self,
        request: &'a RoutingInfo,
        cluster: &'a ClusterState,
    ) -> FallbackPlan<'a> {
        self.default_policy.fallback(request, cluster)
    }

    fn name(&self) -> String {
        format!(
            "RaftLeaderPolicy(fallback={})",
            self.default_policy.name()
        )
    }
}

/// Query system_schema.tables to get the table UUID.
async fn get_table_id(session: &Session, keyspace: &str, table: &str) -> Result<Uuid> {
    let result = session
        .query_unpaged(
            "SELECT id FROM system_schema.tables WHERE keyspace_name = ? AND table_name = ?",
            (keyspace, table),
        )
        .await
        .context("Failed to query system_schema.tables")?;

    let rows = result
        .into_rows_result()
        .context("Expected rows result from system_schema.tables")?;

    let (table_id,): (Uuid,) = rows
        .single_row()
        .context("Expected exactly one row for table_id lookup")?;

    Ok(table_id)
}

/// Information about a single tablet from system.tablets.
struct TabletInfo {
    last_token: i64,
    raft_group_id: Uuid,
    /// Replicas as (host_id, shard) pairs.
    replicas: Vec<(Uuid, i32)>,
}

/// Query system.tablets to get tablet info including replicas for a table.
async fn get_tablet_info(session: &Session, table_id: Uuid) -> Result<Vec<TabletInfo>> {
    // Format the UUID directly in the query since system virtual tables
    // may not support bind markers well.
    let query = format!(
        "SELECT last_token, raft_group_id, replicas FROM system.tablets WHERE table_id = {}",
        table_id
    );

    let result = session
        .query_unpaged(query, ())
        .await
        .context("Failed to query system.tablets")?;

    let rows = result
        .into_rows_result()
        .context("Expected rows result from system.tablets")?;

    let mut tablet_info = Vec::new();
    for row in rows.rows::<(i64, Uuid, Vec<(Uuid, i32)>)>()? {
        let (last_token, raft_group_id, replicas) =
            row.context("Failed to deserialize tablet row")?;
        tablet_info.push(TabletInfo {
            last_token,
            raft_group_id,
            replicas,
        });
    }

    Ok(tablet_info)
}

/// Query the ScyllaDB REST API to get the leader host for a raft group.
async fn get_raft_leader(
    client: &reqwest::Client,
    node: &str,
    port: u16,
    group_id: Uuid,
) -> Result<Uuid> {
    let url = format!("http://{}:{}/raft/leader_host", node, port);

    let resp = client
        .get(&url)
        .query(&[("group_id", group_id.to_string())])
        .send()
        .await
        .with_context(|| {
            format!("Failed to call /raft/leader_host on {}:{}", node, port)
        })?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "/raft/leader_host returned status {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }

    let body = resp.text().await.context("Failed to read response body")?;
    // Response is a JSON string - may be quoted ("uuid") or bare (uuid)
    let trimmed = body.trim().trim_matches('"');
    let uuid = Uuid::parse_str(trimmed).with_context(|| {
        format!(
            "Invalid UUID in /raft/leader_host response: {}",
            body
        )
    })?;

    // A zero UUID means no leader elected yet
    if uuid.is_nil() {
        anyhow::bail!("No leader elected yet for raft group {}", group_id);
    }

    Ok(uuid)
}

/// Extract the hostname/IP from a node address that may include a port.
///
/// Examples:
/// - "127.0.0.1" -> "127.0.0.1"
/// - "127.0.0.1:9042" -> "127.0.0.1"
/// - "hostname" -> "hostname"
/// - "hostname:9042" -> "hostname"
/// - "::1" -> "::1" (IPv6, unchanged)
pub fn extract_host(node: &str) -> &str {
    let colon_count = node.chars().filter(|&c| c == ':').count();
    if colon_count == 1 {
        // "host:port" format
        &node[..node.rfind(':').unwrap()]
    } else {
        // No port or IPv6 address
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_host() {
        assert_eq!(extract_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(extract_host("127.0.0.1:9042"), "127.0.0.1");
        assert_eq!(extract_host("hostname"), "hostname");
        assert_eq!(extract_host("hostname:9042"), "hostname");
        assert_eq!(extract_host("::1"), "::1");
    }

    #[test]
    fn test_find_leader_in_btree() {
        let default_policy =
            scylla::policies::load_balancing::DefaultPolicy::builder().build();
        let policy = RaftLeaderPolicy::new(
            "ks".to_string(),
            "tbl".to_string(),
            default_policy,
        );

        let uuid_a = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let uuid_b = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let uuid_c = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();

        let mut map = BTreeMap::new();
        map.insert(-3000, (uuid_a, 0u32)); // tablet A: ...-3000, shard 0
        map.insert(0, (uuid_b, 1u32)); // tablet B: -2999...0, shard 1
        map.insert(3000, (uuid_c, 2u32)); // tablet C: 1...3000, shard 2

        policy.token_to_leader.set(map).unwrap();

        // Token -5000 -> tablet A (first range)
        assert_eq!(policy.find_leader(-5000), Some((uuid_a, 0)));
        // Token -3000 -> tablet A (boundary)
        assert_eq!(policy.find_leader(-3000), Some((uuid_a, 0)));
        // Token -2999 -> tablet B
        assert_eq!(policy.find_leader(-2999), Some((uuid_b, 1)));
        // Token 0 -> tablet B (boundary)
        assert_eq!(policy.find_leader(0), Some((uuid_b, 1)));
        // Token 1 -> tablet C
        assert_eq!(policy.find_leader(1), Some((uuid_c, 2)));
        // Token 3000 -> tablet C (boundary)
        assert_eq!(policy.find_leader(3000), Some((uuid_c, 2)));
        // Token 5000 -> wraps around to tablet A
        assert_eq!(policy.find_leader(5000), Some((uuid_a, 0)));
    }
}
