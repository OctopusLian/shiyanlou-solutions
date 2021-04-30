#!/bin/bash

#this is a game
#filename : game.sh

function dis_welcome()
{
    declare -r str='0000000010000000000000000000000000000010000100000000000000100000
0000000010000000001000001000000000100001000100000000000000101000
1111110010000000000100110011110000010001000100000000000000100100
0000010011111100000100100010010000010111101111100111111000100100
0000010100000100000000100010010010000010001000000000001000100000
0100100100001000000000100010010001000010010000000000001000111110
0010101001000000111100100010010001000011101111000010010111100000
0001010001000000000100100010010000010010100001000001010000100100
0001000001000000000100100010010000010010100010000000100000100100
0010100010100000000100101011010000100010100010000000100000101000
0010010010100000000100110010100011100010101111100001010000101000
0100010100010000000100100010000000100010100010000001001000010000
1000000100010000000100000010000000100100100010000010001000110010
0000001000001000001010000010000000100100100010000100000001001010
0000010000000100010001111111111000101001101010000000000010000110
0000100000000010000000000000000000010000000100000000000100000010'
    declare -i j=0
    declare -i row=5                                    # 打印起始行
    line_char_count=65                                  # 定义换行位置,每行64个字符加上换行符，一共65个

    echo -ne "\033[37;40m\033[5;3H"                     # 设置颜色和光标起始位置

    for ((i=0; i<${#str}; i++));do
        # 换行
        if [ "$[i%line_char_count]" == "0" ] && [ "$i" != "0" ];then
            row=$row+1
            echo -ne "\033["$row";3H"
        fi
        # 判断前景字符和背景字符
        if [ "${str:$i:1}" == "0" ];then
            echo -ne "\033[37;40m "
        elif [ "${str:$i:1}" == "1" ];then
            echo -ne "\033[31;40m$"
        fi
    done
}

function dismenu()
{
    while [ 1 ]
    do
        draw_border        
        echo -e  "\033[8;30H1) 练习打数字"
        echo -e  "\033[9;30H2) 练习打字母"
        echo -e  "\033[10;30H3) 练习字母数字混合"
        echo -e  "\033[11;30H4) 练习打单词"
        echo -e  "\033[12;30H5) 退出"
        echo -ne "\033[22;2HPlease input your choice: "
        read choice
        case $choice in
        "1" )    draw_border
            #后面两个为函数参数，第一个参数表示打字类型，第二个为移动字符函数
            main digit
            echo -e "\033[39;49m"
            ;;
        "2" )    draw_border
            main char    
            echo -e "\033[39;49m"
            ;;
        "3" )    draw_border
            main mix
            echo -e "\033[39;49m"
            ;;
        "4" )    draw_border
            echo -ne "\033[22;2H"
            read -p "哪个文件你想用于打字游戏练习：" file
            if [ ! -f "$file" ];then
                dismenu
            else
                exec 4<$file # 创建一个文件管道
                main word
                echo -e "\033[39;49m"
            fi
            ;;
        "5"|"q"|"Q" )
            draw_border
            echo -e "\033[10;25Hyou will exit this game, now"
            echo -e "\033[39;49m"
            sleep 1
            clear
            exit 1
            ;;
        * )
            draw_border
            echo -e "\033[22;2Hyour choice is wrong, please try again"    
            sleep 1
            ;;
        esac
    done
}

function draw_border()
{
    declare -i width
    declare -i high
    width=79
    high=23

    clear

    #设置显示颜色,黑底白字
    echo -e "\033[37;40m"
    #设置背景颜色
    for (( i=1; i<=$width; i=i+1 ))
    do        
        for (( j=1; j<=$high; j=j+1))
        do
            #设置显示位置
            echo -e "\033["$j";"$i"H "
        done
    done
    #画背景框
    echo -e "\033[1;1H+\033["$high";1H+\033[1;"$width"H+\033["$high";"$width"H+" 
    for (( i=2; i<=$width-1; i=i+1 ))
    do        
        echo -e "\033[1;"$i"H-"
        echo -e "\033["$high";"$i"H-"
    done
    for (( i=2; i<=$high-1; i=i+1 ))
    do        
        echo -e "\033["$i";1H|"
        echo -e "\033["$i";"$width"H|\n"
    done
}

function doexit()
{  
    draw_border
    echo -e "\033[10;30Hthis game will exit....."
    echo -e "\033[0m"
    sleep 2
    clear
    exit 1
}

#清除整个字符降落区域
function clear_all_area()
{
    local i j
    #填充打字区域
    for (( i=5; i<=21; i++ ))
    do
        for (( j=3; j<=77; j=j+1))
        do
            #设置显示位置
            echo -e "\033[44m\033["$i";"$j"H "
        done
    done
    echo -e "\033[37;40m"
}

# 功能：    清除某一列字符路径
# 传入参数：要清除的那一列的列号
# 返回值：    无
function clear_line()
{
    local i
    #填充打字区域
    for (( i=5; i<=21; i++ ))
    do
        for (( j=$1; j<=$1+9; j=j+1))
        do
            echo -e "\033[44m\033["$i";"$j"H "
        done
    done
    echo -e "\033[37;40m"
}

# 功能：    字符做下落路径移动
# 传入参数：参数1：字符当前行（与字符距超时时间的长度有关）
#             参数2：字符当前的列
#             canshu3:
# 返回值：    无
function move()
{
    
    local locate_row lastloca
    locate_row=$(($1+5))
    #显示需要输入的字符
    echo -e "\033[30;44m\033["$locate_row";"$2"H$3\033[37;40m"
    if [ "$1" -gt "0" ];then
        lastloca=$(($locate_row-1))
        #清楚上一位置
        echo -e "\033[30;44m\033["$lastloca";"$2"H \033[37;40m"    
    fi
}

# 功能：    将相应的不同类型的字符串一次放入数组，用于字符转换
# 传入参数：需要存入数组的字符类型
# 全局变量：array[]
# 返回值：    无
function putarray()
{
    local chars
    case $1 in
    digit) 
        chars='0123456789'
        for (( i=0; i<10; i++ ))
        do
            array[$i]=${chars:$i:1}
        done
        ;;
    char)     
        chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        for (( i=0; i<52; i++ ))
        do
                array[$i]=${chars:$i:1}
        done
        ;;
    mix)     
        chars='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        for (( i=0; i<62; i++ ))
        do
                array[$i]=${chars:$i:1}
        done
        ;;
    *)  
        ;;
    esac    
}

# 功能：    产生相应类型的随机数，用于转换为相应的随机字符
# 传入参数：需要产生的字符类型
# 全局变量：random_char, array[]
# 返回值：    无
function get_random_char()
{
    local typenum
    declare -i typenum=0
    case $1 in
    digit)     
        typenum=$(($RANDOM%10))
        ;;
    char)     
        typenum=$(($RANDOM%52))
        ;;
    mix)     
        typenum=$(($RANDOM%62))
        ;;
    *)
        ;;
    esac    
    random_char=${array[$typenum]}    
}

function main()
{
    declare -i gamestarttime=0
    declare -i gamedonetime=0

    declare -i starttime
    declare -i deadtime
    declare -i curtime
    declare -i donetime
    
    declare -i numright=0
    declare -i numtotal=0
    declare -i accuracy=0

    #将相应的字符存进数组,$1为用户选择的击打字符类型
    putarray $1

    # 初始化游戏开始时间
    gamestarttime=`date +%s`

    while [ 1 ]
    do                        
        echo -e "\033[2;2H请输入屏幕中的字母，在字符消失前！"

        echo -e "\033[3;2H游戏时间:     "
        curtime=`date +%s`
        gamedonetime=$curtime-$gamestarttime    
        echo -e "\033[31;40m\033[3;15H$gamedonetime s\033[37;40m"
        echo -e "\033[3;60H总数：\033[31;26m$numtotal\033[37;40m"
        echo -e "\033[3;30H正确率：\033[31;40m$accuracy %\033[37;40m"
        echo -ne "\033[22;2H你的输入为：                         "
        clear_all_area
        # 循环10次检查某一列字符是否超时或是否被击落
        for (( line=20; line<=60; line=line+10 ))
        do
            #检查该列字符是否被击落        
            if [ "${ifchar[$line]}" == "" ] || [ "${donetime[$line]}" -gt "$time" ]
            then
                # 清除该列显示
                clear_line $line
                #重新产生一个随机字符
                if [ "$1" == "word" ];then
                    read -u 4 word
                    if [ "$word" == "" ];then # 文件读取结束
                        exec 4<$file
                    fi
                    putchar[$line]=$word
                else
                    get_random_char $1
                    putchar[$line]=$random_char
                fi
                numtotal=$numtotal+1
                # 标志位，置1
                ifchar[$line]=1
                #重新设置计时器
                starttime[$line]=`date +%s`
                curtime[$line]=${starttime[$line]}
                donetime[$line]=$time
                #重新初始字符位置行为0
                column[$line]=0
                if [ "$1" == "word" ];then
                    move 0 $line ${putchar[$line]}
                fi
            else
                #没有超时或被击落，则更新计时器和当前位置
                curtime[$line]=`date +%s`
                donetime[$line]=${curtime[$line]}-${starttime[$line]}
                move ${donetime[$line]} $line ${putchar[$line]}
            fi
        done

        if [ "$1" != "word" ];then
            echo -ne "\033[22;14H" # 清空输入行字符
            # 检查用户输入字符,同时作为一秒定时器
            if read -n 1 -t 0.5 tmp
            then
                # 成功读入，循环检查输入是否与某一列匹配
                for (( line=20; line<=60; line=line+10 ))
                do
                    if [ "$tmp" == "${putchar[$line]}" ];then
                        # 清除该列显示
                        clear_line $line
                        # 如果匹配，清除标志位
                        ifchar[$line]=""
                        echo -e "\007\033[32;40m\033[4;62H         right !\033[37;40m"
                        numright=$numright+1
                        # 退出循环
                        break
                    else
                        # 否则始终显示错误，直到超时                    
                        echo -e "\033[31;40m\033[4;62Hwrong,try again!\033[37;40m"
                    fi
                done
            fi
        else
            echo -ne "\033[22;14H" # 清空输入行字符
            # 检查用户输入字符,同时作为定时器
            if read tmp
            then
                # 成功读入，循环检查输入是否与某一列匹配
                for (( line=20; line<=60; line=line+10 ))
                do
                    if [ "$tmp" == "${putchar[$line]}" ];then
                        # 清除该列显示
                        clear_line $line
                        # 如果匹配，清除标志位
                        ifchar[$line]=""
                        echo -e "\007\033[32;40m\033[4;62H         right !\033[37;40m"
                        numright=$numright+1
                        # 退出循环
                        break
                    else
                        # 否则始终显示错误，直到超时                    
                        echo -e "\033[31;40m\033[4;62Hwrong,try again!\033[37;40m"
                    fi
                done
            fi
        fi
        trap " doexit " 2 # 捕获特殊信号
        # 计算正确率
        accuracy=$numright*100/$numtotal
    done
}

#定义打字超时时间变量，难度选择既是设定不同的打字超时时间
declare -i time
function modechoose()
{
    echo -e  "\033[8;30H1) easy mode"
    echo -e  "\033[9;30H2) normal mode"
    echo -e  "\033[10;30H3) difficult mode"
    echo -ne "\033[22;2HPlease input your choice: "
    read mode    
    case $mode in
        "1" )
            time=10
            dismenu # 调用菜单选择函数
            ;;
        "2" )
            time=5
            dismenu
            ;;
        "3" )
            time=3
            dismenu
            ;;
        * )
            echo -e "\033[22;2Hyour choice is wrong, please try again"    
            sleep 1
    esac
}

# -----------------------------------------------------------------
# 主程序流程


draw_border
dis_welcome
echo -ne "\033[3;30Hstart the game. Y/N : "
read yourchoice
if [ "$yourchoice" == "Y" ] || [ "$yourchoice" == "y" ];then
draw_border
modechoose
else 
    clear
    exit 1
fi

exit 1
