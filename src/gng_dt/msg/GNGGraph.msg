# GNGGraph.msg
# GNGグラフ全体

# ヘッダ
std_msgs/Header header

# ノードリスト
GNGNode[] nodes

# 統計情報
int32 total_nodes
int32 traversable_nodes
int32 attention_nodes