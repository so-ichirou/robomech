# GNGNode.msg
# 単一のGNGノード情報

# 位置
float64 x
float64 y
float64 z

# 法線
float64 nx
float64 ny
float64 nz

# 属性
bool traversability
bool has_attention
int32 attention_count

# ID
int32 node_id

# 隣接ノードID（POSITIONトポロジー）
int32[] neighbor_ids