import csv
input_path = 'town5_waypoints.csv'   # 原始文件路径
output_path = 'town5_waypoints2.csv'   # 也可以用新的名字保存，比如 'filtered_path.csv'# 读取原始数据
with open(input_path, 'r', newline='') as infile:
  rows = list(csv.reader(infile))# 删除第2到140行（索引1~139）
filtered_rows = [rows[0]] + rows[160:] # 保留header + 后面的数据# 写回文件
with open(output_path, 'w', newline='') as outfile:
  writer = csv.writer(outfile)
  writer.writerows(filtered_rows)
  print(f"已删除第2到140行，写入到：{output_path}")