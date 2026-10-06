import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/thienhoang/Documents/HRI/4/ur3_llm_ws/install/ur3_llm_control'
