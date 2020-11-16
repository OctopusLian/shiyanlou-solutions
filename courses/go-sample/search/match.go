package search

import (
	"log"
)

// Result保存搜索的结果
type Result struct {
	Field   string
	Content string
}

// Matcher定义了要实现的新搜索类型的行为
/*
interface 关键字声明了一个接口，这个接口声明了结构类型或者具名类型需要实现的行为。一个接口的行为最终由在这个接口类型中声明的方法决定。
命名接口的时候，也需要遵守 Go 语言的命名惯例。如果接口类型只包含一个方法，那么这个类型的名字以 er 结尾。
如果接口类型内部声明了多个方法，其名字需要与其行为关联。
如果要让一个用户定义的类型实现一个接口，这个用户定义的类型要实现接口类型里声明的所有方法。
*/
type Matcher interface { //声明了一个名为 Matcher 的接口类型
	/*
	   对于 Matcher 这个接口来说，只声明了一个 Search 方法，这个方法输入一个指向 Feed 类型值的指针和一个 string 类型的搜索项。
	   这个方法返回两个值：一个指向 Result 类型值的指针的切片，另一个是错误值。
	*/
	Search(feed *Feed, searchTerm string) ([]*Result, error)
}

// Match函数，为每个数据源单独启动goroutine来执行这个函数，并发地执行搜索
/*
这个函数使用实现了 Matcher 接口的值或者指针，进行真正的搜索。
这个函数接受 Matcher 类型的值作为第一个参数。只有实现了 Matcher 接口的值或者指针能被接受。
因为 defaultMatcher 类型使用值作为接收者，实现了这个接口，所以 defaultMatcher 类型的值或者指针可以传入这个函数。
*/
func Match(matcher Matcher, feed *Feed, searchTerm string, results chan<- *Result) {
	// 对特定的匹配器执行搜索
	searchResults, err := matcher.Search(feed, searchTerm)
	if err != nil {
		log.Println(err)
		return
	}

	// 对特定的匹配器执行搜索
	for _, result := range searchResults {
		results <- result //把结果写入通道，以便正在监听通道的 main 函数就能收到这些结果
	}
}

// Display从每个单独的goroutine接收到结果后,在终端窗口输出
func Display(results chan *Result) {
	// 通道会一直被阻塞，直到有结果写入
	// 一旦通道被关闭，for循环就会终止
	/*
		当通道被关闭时，通道和关键字 range 的行为，使这个函数在处理完所有结果后才会返回。
	*/
	for result := range results {
		log.Printf("%s:\n%s\n\n", result.Field, result.Content)
	}
}
