var Promise = require("bluebird");


/**
 * 保留字符在一行的一个对象
 * @type {{}}
 */
module.exports = {
    /**
     * 把需要保留在一行的字符串从整个字符串中抠出来，用占位符替换
     * @param str
     * @returns {*}
     */
    pickFromCode: function (str) {
        return pickFromCode(str);
    },
    result: result,
    /**
     * 从pickFromCode把不能分离的行替换回来，另外对字符串以5个字符为单位做了分离处理，防止字符串太长影响效果。
     * @param lines
     */
    findBack: function (lines) {
        return findBack(lines);
    }

}
var result = {}
/**
 把字符串中的字符串正则都提取出来，用占位符表示。之后可以替换回来
 */
var PREFIX = "pic_and_code_to_ascii_"
var placeholderIndex = 0;
var double_operator = ["==", ">=", "<=", "+=", "-=", "*=", "/=", "%=", "++", "--", "&&", "||", ">>", "<<"]
var three_operator = ['===', '!==']

var other_operator = ['return function', 'throw new Error']
/**
 return o.call(e,t)
 return!c(n,e)
 return h.call(e)==="[object Array]"
 return typeof e==="function"
 return String(e).replace(/[&<>"'\/]/g,function(e){return v[e]})
 return[new RegExp(p(e[0]
 return this.tail===""
 return"script";


 return[^a-zA-Z_0-1][a-zA-Z_0-1]+
 */
//根据正则保留在一行。
var reg_operator = [
    {
        start: "return\"",
        reg: /^return".*?"/ // return "d" 或者 return ""
    },
    {
        start: "return\'",
        reg: /^return'.*?'/  // return 'd' 或者 return ''
    }, {
        start: "return\/",
        reg: /^return\/.+?\/[igm]*/  // return//g 或者 return ''
    },
    {
        start: "return",
        reg: /^return[^a-zA-Z_0-1"'};][a-zA-Z_0-1.]+/ // return 0.1 或者 return function 或者return aaabb
    },
    {
        start: "throw",
        reg: /^throw [a-zA-Z_0-1]+?/ //throw new 或者 throw obj
    }
]
//向前补全 ddd++ 的情况
var findPrevNotABC = function (str, index) {
    var i = index;
    var s;
    do {
        i--;
        s = str.charAt(i);
    } while(/[a-zA-Z_0-9]/.test(s))//当前取得 s 字符如果匹配变量命名标准，那么就继续往前取，直到把这个变量取完，返回变量开始的下标 i
    return i + 1;
}

function createPlaceholder () {
    return " " + PREFIX + (placeholderIndex++) + " ";
}

function pickFromCode (str) {

    //从代码字符串里把所有引号里的字符串和正则抠出来。
    var pickStart_double = 0;
    var pickEnd_double = 0;
    var pickStart_single = 0;
    var pickEnd_single = 0;
    var pickStart_reg = 0;
    var pickEnd_reg = 0;
    var is_in_double_quot = false;
    var is_in_single_quot = false;
    var is_in_reg = false;
    var is_in_other = false;
    var pickStart_other = 0;
    var pickEnd_other = 0;
    //抠出来放到 strs 里面
    var strs = []

    //进入状态机
    //从第一个字符开始遍历，一个一个遍历
    for(var i = 0; i < str.length; i++) {
        var now_char = str[i];
        //记录前前四个字符
        var last_char = i > 0 ? str[i - 1] : null;
        var last_two_char = i > 1 ? str[i - 2] : null;
        var last_three_char = i > 2 ? str[i - 3] : null;
        var last_four_char = i > 3 ? str[i - 4] : null;
        //记录后三个字符
        var next_char = i < (str.length - 1) ? str[i + 1] : null;
        var next_two_char = i < (str.length - 2) ? str[i + 2] : null;
        var next_three_char = i < (str.length - 3) ? str[i + 3] : null;


        /**
         * 不在正则，不在引号之内，检查是否存在operator
         */
        if(!is_in_reg && !is_in_double_quot && !is_in_single_quot) {

            //是否存在三目运算符的 operator
            if(three_operator.indexOf(now_char + next_char + next_two_char) != -1) {
                //存在  记录起始位置和结束为止
                pickStart_other = i;
                pickEnd_other = i + 2;
                //遍历进入到三目运算符结尾处，即 i += 2
                i += 2;
                //把这个运算符抠出来用对象的形式放到 strs 里面去。
                strs.push({
                    value: str.substring(pickStart_other, pickEnd_other + 1),
                    type: "other",
                    start: pickStart_other,
                    end: pickEnd_other
                })
                continue;
                //是否存在双目运算符
            } else if(double_operator.indexOf(now_char + next_char) != -1) {
                pickStart_other = i;
                pickEnd_other = i + 1;
                //如果是 ++ 或者 -- 还要找到前面的变量，它们也必须放在一行
                if(now_char + next_char == "++" || now_char + next_char == "--") {
                    pickStart_other = findPrevNotABC(str, i);
                } else {

                }
                //跳到双目运算符的结尾处
                i += 1;
                strs.push({
                    value: str.substring(pickStart_other, pickEnd_other + 1),
                    type: "other",
                    start: pickStart_other,
                    end: pickEnd_other
                })
                continue;
            }
            //根据正则保留的处理 return 的各种形式和 throw
            reg_operator.forEach(function (o) {
                var start = o.start;
                var reg = o.reg;
                var s = str.substr(i, start.length)
                //匹配到 return 或是 throw 开头的部分
                if(s == start) {
                    //符合此正则，进入正则判断逻辑
                    //从当前位置到 str 整个 code 结束，赋值给 sub，在 sub 里找是否有匹配 reg 的
                    var sub = str.substring(i, str.length - 1);
                    var match = sub.match(reg);
                    //如果有，不一定只有一个，可能有多个，但是我们只处理第一个，其它的下次遇到在处理
                    if(match) {
                        //m 是匹配到的第一个
                        var m = match[0];
                        //截取匹配到的部分，比较一下是否与字符串截取到的一致
                        var s = str.substr(i, m.length)
                        if(s == m) {
                            //一致则 push 到 strs 里面去。
                            pickStart_other = i;
                            pickEnd_other = i + m.length - 1;
                            i += m.length - 1;
                            strs.push({
                                value: str.substring(pickStart_other, pickEnd_other + 1),
                                type: "other",
                                start: pickStart_other,
                                end: pickEnd_other
                            })
                        }
                    }
                }
            })

            //处理小数点。 0.11 11.2233
            if(now_char == ".") {
                //往前找数字
                var prev_nums = [];
                var nowI = i;
                nowI--;
                var c = nowI > 0 ? str[nowI] : null;
                while(/[0-9]/.test(c)) {
                    //往前找数字，是数字就往 prev_nums 数组的开头添加这个新元素
                    prev_nums.unshift(c);
                    //然后继续往前找，直到找到属于这小数点的所有数字
                    nowI--;
                    c = nowI > 0 ? str[nowI] : null;
                }
                //往后找数字
                var next_nums = [];
                var nowI = i;
                //往后找
                nowI++;
                //如果没到末尾，就取字符
                var c = nowI < (str.length - 1) ? str[nowI] : null;
                //如果是数字
                while(/[0-9]/.test(c)) {
                    //在数组末尾添加这个数字
                    next_nums.push(c);
                    //继续找
                    nowI++;
                    //赋值
                    c = nowI < (str.length - 1) ? str[nowI] : null;
                }
                //小数点前后有数，即往前的和往后的数字至少有一个不为空
                if(prev_nums.length || next_nums.length) {
                    //把这个数字的开头和结尾找出来
                    var start = i - prev_nums.length;
                    var end = i + next_nums.length;
                    //抠出来，压到数组里准备替换掉
                    strs.push({
                        value: str.substring(start, end + 1),
                        type: "other",
                        start: start,
                        end: end
                    })
                }

            }

        }
        //判断是正则表达式的开始
        if(!is_in_single_quot && !is_in_double_quot) {
            if(!is_in_reg) {
                if(now_char == "/" && last_char != "<" && !/[0-9a-zA-Z_)\]]/.test(last_char)) {
                    //探测到字符串引号出现，首先判断不是转义的。
                    if(last_char != "\\" && last_char != "\"" && last_char != "\'") {
                        pickStart_reg = i;
                        is_in_reg = true;
                    }
                }
            } else {
                //如果现在在正表达式里
                if(now_char == "/") {
                    //探测到结尾的斜杠出现，首先判断不是转义的。
                    //三种情况，1.前一个字符不是转义反斜杠；
                    //2.前一个字符是转义符，再往前一个又是转义符，抵消了，再往前不是；
                    //3.前四个都是转义符，抵消了
                    if(last_char != "\\" ||
                      (last_char == "\\" && last_two_char == "\\" && last_three_char != "\\")
                      || (last_char == "\\" && last_two_char == "\\" && last_three_char == "\\" && last_four_char == "\\")) { //
                        //正则表达式结尾的地方，序号i 赋值给 pickEnd
                        pickEnd_reg = i;
                        is_in_reg = false;
                        //往后找 是否存在 gim  这个g也算是正则表达式的部分，要把正则表达式的结尾延长
                        if("gim".indexOf(next_char) != -1) {
                            pickEnd_reg++;
                            if("gim".indexOf(next_two_char) != -1) {
                                pickEnd_reg++;
                                if("gim".indexOf(next_three_char) != -1) {
                                    pickEnd_reg++;
                                }
                            }
                        }
                        //将找到的正则表达式填入 strs
                        strs.push({
                            value: str.substring(pickStart_reg, pickEnd_reg + 1),
                            type: "reg",
                            start: pickStart_reg,
                            end: pickEnd_reg
                        })
                    }
                }
            }
        }
        //判断出现双引号，且前面的字符不是转义字符，确定当前进入双引号的字符串
        if(!is_in_single_quot && !is_in_reg) {
            if(!is_in_double_quot) {
                if(now_char == "\"") {
                    //探测到字符串引号出现，首先判断不是转义的。
                    if(last_char != "\\") {
                        pickStart_double = i;
                        is_in_double_quot = true;
                    }
                }
            } else {
                //找到了双引号字符串结尾的双引号
                if(now_char == "\"") {
                    //这里只有两种情况来判断不是转义的双引号
                    if(last_char != "\\" || (last_char == "\\" && last_two_char == "\\" && last_three_char != "\\")) {
                        pickEnd_double = i;
                        is_in_double_quot = false;
                        //push 到 strs 里
                        strs.push({
                            value: str.substring(pickStart_double, pickEnd_double + 1),
                            type: "double",
                            start: pickStart_double,
                            end: pickEnd_double
                        })
                    }
                }
            }
        }
        //找到单引号
        if(!is_in_double_quot && !is_in_reg) {
            if(!is_in_single_quot) {
                if(now_char == "\'") {
                    //探测到字符串引号出现，首先判断不是转义的。
                    if(last_char != "\\") {
                        pickStart_single = i;
                        is_in_single_quot = true;
                    }
                }
            } else {
                //如果现在在字符串里
                if(now_char == "\'") {
                    //探测到字符串引号出现，首先判断不是转义的。
                    if(last_char != "\\" || (last_char == "\\" && last_two_char == "\\" && last_three_char != "\\")) {
                        pickEnd_single = i;
                        is_in_single_quot = false;
                        strs.push({
                            value: str.substring(pickStart_single, pickEnd_single + 1),
                            type: "single",
                            start: pickStart_single,
                            end: pickEnd_single
                        })
                    }
                }
            }
        }

    }
    //处理下，把str中的相应部分都替换成占位符
    var str_result = "";
    var start_offset = 0;
    result = {}
    for(var i in strs) {
        var s = strs[i];
        var placehoder = createPlaceholder();
        //填充到 str ，str_result 就是分割 str，
        str_result += str.substring(start_offset, s.start);
        str_result += placehoder;
        s.placehoder = placehoder;
        //然后起始位置移动到被替换部分的结尾处
        start_offset = s.end + 1;
        //记录被替换的内容和对应占位符的关系，方便替换回来
        result[placehoder] = s;
    }
    // 所有分割的计数了，再加上从最后一次替换到代码结尾部分的内容
    str_result += str.substring(start_offset, str.length);
    return str_result;
}


var findBack = function (lines) {
    //lines 是被替换后的代码，是一个数组，现在要对代码进行遍历，替换回来
    for(var i = 0; i < lines.length; i++) {
        var line = lines[i];
        //取数组的元素
        if(line.indexOf(PREFIX) != -1) {
            //当前取出的元素是占位符
            var line_data = result[" " + line + " "];//找回占位符的数据
            if(line_data.type == "double") {
                //往后数三个元素，引号中间的字符串是一个元素，数三个可以数到冒号
                if(lines[i + 3] == ":") {
                    //如果是 "dd":"dd" 不处理
                    lines.splice(i, 1, line_data.value)//删除 i 位置开始的 1 个元素，并用 line_data.value 替换
                } else {
                    var arr = splitDoubleQuot(line_data.value); //拆分一下字符串
                    lines.splice(i, 1);///先删掉
                    arr.forEach(function (a, n) {
                        lines.splice(i + n, 0, a) //用 forEach 循环把数据插回去
                    })
                }


            } else if(line_data.type == "single") {
                // var arr = self.splitSingleQuot(line_data.value); //拆分一下字符串
                if(lines[i + 3] == ":") {
                    //如果是 "dd":"dd" 不处理
                    lines.splice(i, 1, line_data.value)
                } else {
                    var arr = splitSingleQuot(line_data.value); //拆分一下字符串
                    lines.splice(i, 1);
                    arr.forEach(function (a, n) {
                        lines.splice(i + n, 0, a) //把数据插回去
                    })
                }
            } else {
                //其他情况，不用或者不能对替换回去的字符串进行拆分的情况，比如正则表达式等
                lines.splice(i, 1, line_data.value)
            }
        }
    }
}

var splitDoubleQuot = function (str) {
    str = str.replace(/\\\\/g, "☃")
    var r = [];
    //五个五个的
    var s = str.substring(1, str.length - 1)
    var len = s.length;
    r.push("(")
    var last_cursor = 0;

    while(last_cursor < len) {
        var l = 5;
        //一次分割 5 个字符
        //处理 l 如果分割字符串的时候，分割到了不可分割的字符，就延长
        if(s.charAt(last_cursor + l - 1) == "\\") { //处理\a
            l++;
            if(s.charAt(last_cursor + l - 1) == "\\") { //处理 \\\
                l++;
            }
        }
        //处理\x0a
        if(s.charAt(last_cursor + l - 1) == "x" && s.charAt(last_cursor + l - 2) == "\\") {
            l += 2;
        }
        //每一组分割的字符串为 n
        var n = s.substring(last_cursor, last_cursor + l)
        last_cursor = last_cursor + n.length;
        n = n.replace(/☃/g, "\\\\");
        r.push("\"" + n + "\"")

        r.push("+")
    }

    if(len == 0) {
        r.push('""')
    } else {
        //多出的一个 + 号是在这里 pop 掉的
        r.pop()
    }
    r.push(")")
    return r;
}
var splitSingleQuot = function (str) {
    str = str.replace(/\\\\/g, "☃")
    var r = [];
    //把字符串分成三个三个的。例如"abcd" 编程 "acb"+"d"
    var s = str.substring(1, str.length - 1)
    var len = s.length;
    var cut_len = Math.ceil((len) / 5);
    r.push("(")
    var last_cursor = 0;

    while(last_cursor < len) {
        var l = 5;
        if(s.charAt(last_cursor + l - 1) == "\\") { //处理\a
            l++;
            if(s.charAt(last_cursor + l - 1) == "\\") { //处理 \\\
                l++;
            }
        }
        if(s.charAt(last_cursor + l - 1) == "x" && s.charAt(last_cursor + l - 2) == "\\") {
            l += 2;
        }
        var n = s.substring(last_cursor, last_cursor + l)
        last_cursor = last_cursor + n.length;
        n = n.replace(/☃/g, "\\\\");
        r.push("\'" + n + "\'")


        r.push("+")
    }

    if(len == 0) {
        r.push("''")
    } else {
        r.pop()
    }
    r.push(")")
    return r;
}