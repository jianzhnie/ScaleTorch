# 查找空节点
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh /home/jianzhnie/llmtuner/llm/MindSpeed-LLM/scripts/nodes/node_list1.txt
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh /home/jianzhnie/llmtuner/llm/MindSpeed-LLM/scripts/nodes/node_list3.txt
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh /home/jianzhnie/llmtuner/llm/MindSpeed-LLM/scripts/nodes/node_list_all.txt

# 停止所有节点
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh available_nodes.txt
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh node_list.txt
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh /home/jianzhnie/llmtuner/llm/MindSpeed-LLM/scripts/nodes/node_list_all.txt

bash ./scripts/torch_dist/launch_multi_nodes.sh