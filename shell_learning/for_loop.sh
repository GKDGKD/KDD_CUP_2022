names=("John" "Mike" "Sarah" "Kate" "Sam")  # 注意 = 前后不要加空格
ages=(25 34 28 27 32)  # 创建数组

# 获取数组全部元素：使用 ${array[@]} 来打印数组的全部元素。这个语法会展开数组，并将每个元素作为独立的参数传递给命令。
echo "names: ${names[@]}"
echo "ages: ${ages[@]}"

first_name="${names[0]}"  # 获取数组的第一个元素
echo "First name: $first_name"

# 打印数组长度
echo "ages length: ${#ages[@]}"

# 使用for循环打印每个人的名字和年龄
for ((i=0; i<${#names[@]}; i++)); 
do
    echo "Name: ${names[$i]}, Age: ${ages[$i]}"
done