package search //这个代码文件在 search 文件夹里，所以第 01 行声明了包的名字为 search。

import (
	"encoding/json" //json 包提供编解码 JSON 的功能
	"os"            //os 包提供访问操作系统的功能，如读文件。
)

/*
因为 Go 编译器可以根据赋值运算符右边的值来推导类型，声明常量的时候不需要指定类型。
此外，这个常量的名称使用小写字母开头，表示它只能在 search 包内的代码里直接访问，而不暴露到包外面。
*/
const dataFile = "data/data.json" //声明了一个叫作 dataFile 的常量，使用内容是磁盘上根据相对路径指定的数据文件名的字符串做初始化。

// Feed 包含我们需要处理的数据源的信息
/*
声明了一个名叫 Feed 的结构类型。这个类型会对外暴露。这个类型里面声明了 3 个字段，每个字段的类型都是字符串，对应于数据文件中各个文档的不同字段。
每个字段的声明最后 ` 引号里的部分被称作标记（tag）。这个标记里描述了 JSON 解码的元数据，用于创建 Feed 类型值的切片。
每个标记将结构类型里字段对应到 JSON 文档里指定名字的字段。
*/
type Feed struct {
	Name string `json:"site"`
	URI  string `json:"link"`
	Type string `json:"type"`
}

// RetrieveFeeds读取并反序列化源数据文件
/*
函数功能：这个函数读取数据文件，并将每个 JSON 文档解码，存入一个 Feed 类型值的切片里。
函数没有参数，会返回两个值。第一个返回值是一个切片，其中每一项指向一个 Feed 类型的值。第二个返回值是一个 error 类型的值，用来表示函数是否调用成功。
*/
func RetrieveFeeds() ([]*Feed, error) {
	// 打开文件
	file, err := os.Open(dataFile) //使用相对路径调用 Open 方法，并得到两个返回值。第一个返回值是一个指针，指向 File 类型的值，第二个返回值是 error 类型的值，检查 Open 调用是否成功。
	if err != nil {
		return nil, err
	}

	// 当函数返回时
	// 关闭文件
	/*
		关键字 defer 会安排随后的函数调用在函数返回时才执行。在使用完文件后，需要主动关闭文件。
		使用关键字 defer 来安排调用 Close 方法，可以保证这个函数一定会被调用。
		哪怕函数意外崩溃终止，也能保证关键字 defer 安排调用的函数会被执行。
		关键字 defer 可以缩短打开文件和关闭文件之间间隔的代码行数，有助提高代码可读性，减少错误。
	*/
	defer file.Close()

	// 将文件解码到一个切片里
	// 这个切片的每一项是一个指向一个Feed类型值的指针
	var feeds []*Feed //声明了一个名字叫 feeds，值为 nil 的切片，这个切片包含一组指向 Feed 类型值的指针。
	/*
		调用 json 包的 NewDecoder 函数，然后在其返回值上调用 Decode 方法。
		我们使用之前调用 Open 返回的文件句柄调用 NewDecoder 函数，并得到一个指向 Decoder 类型的值的指针。
		之后再调用这个指针的 Decode 方法，传入切片的地址。之后 Decode 方法会解码数据文件，并将解码后的值以 Feed 类型值的形式存入切片里。
		根据 Decode 方法的声明，该方法可以接受任何类型的值,
		Decode 方法接受一个类型为 interface{} 的值作为参数。这个类型在 Go 语言里很特殊，一般会配合 reflect 包里提供的反射功能一起使用。
	*/
	err = json.NewDecoder(file).Decode(&feeds)

	// 这个函数不需要检查错误，调用者会做这件事
	return feeds, err //函数的调用者返回了切片和错误值
}
