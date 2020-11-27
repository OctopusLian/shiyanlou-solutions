先导入 Pillow 模块和 sys 模块：

根据前文所述实验原理创建一个函数，用于将图片的像素点数据值的二进制末位全部变成 0 ，以便后续事宜末位存储数据。该函数命名为 make_even_image ，它接收一个图片对象作为参数，返回一个处理后的图片对象：

def make_even_image(image):

图片处理功能写好之后，就可以向图片对象中写入信息了，也就是把字符串编码到图片中。完成这一功能的函数为 encode_data_in_image ，该函数接收两个参数：图片对象和要隐藏到图片中的信息，函数的返回值是隐藏了字符串信息的新图片。函数代码如下：

def encode_data_in_image(image, data):

相关函数的文档链接：

[Image.getdata()](https://pillow.readthedocs.io/en/3.3.x/reference/Image.html#PIL.Image.Image.getdata)  
[PIL.Image.new()](https://pillow.readthedocs.io/en/3.3.x/reference/Image.html#PIL.Image.new)  
[PIL.Image.Image.putdata()](https://pillow.readthedocs.io/en/3.3.x/reference/Image.html#PIL.Image.Image.putdata)  

encode_data_in_image 函数中，bytearray 方法将字符串转换为整数值序列（数字范围是 0 到 2^8-1），数值序列由字符串的字节数据转换而来，如下图：  

[](./1.png) 

utf-8 编码的一个中文字符就占了 3 个字节，那么四个字符共占 3×4=12 个字节，于是共有 12 个数值。然后 map(int_to_binary_str, bytearray(data, 'utf-8')) 对数值序列中的每一个值应用 int_to_binary_str 匿名函数，将十进制数值序列转换为二进制字符串序列。匿名函数里 bin 方法的作用是将一个 int 值转换为二进制字符串，详见： https://docs.python.org/3/library/functions.html#bin  


通过前面的学习，我们已经可以悄无声息地将字符串隐藏到图片中，接下来就从图片中解析出隐藏的信息。  

首先，从图片对象的像素点数据中提取各数值的最末位，然后每 8 个一组，就是一个字节的数据，但每个字符对应的字节数不定，可能是一二三四中的任意数。  

定义 decode_data_from_image 函数用于从图片对象中解析得到字符串，显然该函数的参数为隐藏了信息的图片对象，返回值是解析得到的字符串：  

def decode_data_from_image(image):  

找到数据截止处所用的字符串 '0000000000000000' 很有意思，它的长度为16，而不是直觉上的 8，因为两个包含数据的字节的接触部分可能有 8 个 0。  

其中用到了一个 binary_to_string 函数，该函数用于将二进制字符串处理得到字符。其代码如下：  

def binary_to_string(binary):  

要理解这个函数的原理，必须要先搞懂 UTF-8 编码的方式，可以在 wikipedia 上了解 UTF-8 编码：https://zh.wikipedia.org/wiki/UTF-8  

UTF-8 是 UNICODE 的一种变长度的编码表达方式，也就是说一个字符串中，不同的字符所占的字节数不一定相同，这就给我们的工作带来了一点复杂度，如果我们要支持中文的话。
码点的位数 	码点起值 	码点终值 	字节序列 	Byte 1 	Byte 2 	Byte 3 	Byte 4 	Byte 5 	Byte 6
7 	U+0000 	U+007F 	1 	0xxxxxxx 					
11 	U+0080 	U+07FF 	2 	110xxxxx 	10xxxxxx 				
16 	U+0800 	U+FFFF 	3 	1110xxxx 	10xxxxxx 	10xxxxxx 			
21 	U+10000 	U+1FFFFF 	4 	11110xxx 	10xxxxxx 	10xxxxxx 	10xxxxxx 		
26 	U+200000 	U+3FFFFFF 	5 	111110xx 	10xxxxxx 	10xxxxxx 	10xxxxxx 	10xxxxxx 	
31 	U+4000000 	U+7FFFFFFF 	6 	1111110x 	10xxxxxx 	10xxxxxx 	10xxxxxx 	10xxxxxx 	10xxxxxx

在上图中，只有 x 所在的位置（也即是字节中第一个 0 之后的数据）存储的是真正的字符数据，因此我们使用下面这个嵌套函数来提取数据：

def effective_binary(binary_part, zero_index):  

该函数接收两个参数：二进制字符串片段（该片段的字符数量一定是 8 的整倍数）和字节数。它会在内部每 8 个字符为一个字节进行处理，得到其中的有效位，最后将有效位连起来返回。  

string = chr(int(effective_binary(binary[index: index+length], zero_index), 2)) 这行代码中用到了两个方法 chr 和 int ，它们的说明如下：  

    int ：接受两个参数，第一个参数为数字字符串，第二个参数为这个数字字符串代表的数字的进制。详见： https://docs.python.org/3/library/functions.html#int  

    chr ：接受一个参数，参数为 int 值，返回 Unicode 码点为这个 int 值的字符。while 循环的最后我们将当前字符的索引增加当前字符的长度，得到下一个字符的索引。  

通过 decode_data_from_image 函数在内部调用 binary_to_string 函数，最后得到了隐藏在图片对象中的完整信息。  

前面已经定义了向图片中隐藏信息和从图片中解析信息的方法，最后定义一个 main 函数来测试功能即可。  