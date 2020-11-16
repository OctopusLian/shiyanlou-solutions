package search

import (
	"log"  //log 包提供打印日志信息到标准输出（stdout）、标准错误（stderr）或者自定义设备的功能。
	"sync" //sync 包提供同步 goroutine 的功能。
)

// 注册用于搜索的匹配器的映射
/*
Notes:
map 是 Go 语言里的一个引用类型，需要使用 make 来构造。
如果不先构造 map 并将构造后的值赋值给变量，会在试图使用这个 map 变量时收到出错信息。
这是因为 map 变量默认的零值是 nil。
在 Go 语言中，所有变量都被初始化为其零值。对于数值类型，零值是 0；对于字符串类型，零值是空字符串；对于布尔类型，零值是 false；
对于指针，零值是 nil。
对于引用类型来说，所引用的底层数据结构会被初始化为对应的零值。但是被声明为其零值的引用类型的变量，会返回 nil 作为其值。
*/
var matchers = make(map[string]Matcher)

// Run执行搜索逻辑
/*
Go 语言使用关键字 func 声明函数，关键字后面紧跟着函数名、参数以及返回值。
*/
func Run(searchTerm string) {
	// 获取需要搜索的数据源列表
	/*
		这个函数返回两个值。第一个返回值是一组 Feed 类型的切片。切片是一种实现了一个动态数组的引用类型。第二个返回值是一个错误值。

		简化变量声明运算符（:=）。这个运算符用于声明一个变量，同时给这个变量赋予初始值。
		编译器使用函数返回值的类型来确定每个变量的类型。简化变量声明运算符只是一种简化记法，让代码可读性更高。
		这个运算符声明的变量和其他使用关键字 var 声明的变量没有任何区别。
	*/
	feeds, err := RetrieveFeeds()
	if err != nil {
		/*
			检查返回的值是不是真的是一个错误。如果真的发生错误了，就会调用 log 包里的 Fatal 函数。
			Fatal 函数接受这个错误的值，并将这个错误在终端窗口里输出，随后终止程序。
		*/
		log.Fatal(err)
	}

	// 创建一个无缓冲的通道，接收匹配后的结果
	/*
		在 Go 语言中，通道（channel）和映射（map）与切片（slice）一样，也是引用类型，不过通道本身实现的是一组带类型的值，这组值用于在 goroutine 之间传递数据。
		通道内置同步机制，从而保证通信安全。
	*/
	results := make(chan *Result)

	/*
		在 Go 语言中，如果 main 函数返回，整个程序也就终止了。Go 程序终止时，还会关闭所有之前启动且还在运行的 goroutine。
		写并发程序的时候，最佳做法是，在 main 函数返回前，清理并终止所有之前启动的 goroutine。
		编写启动和终止时的状态都很清晰的程序，有助减少 bug，防止资源异常。
	*/

	// 构造一个waitGroup，以便处理所有的数据源
	/*
		这个程序使用 sync 包的 WaitGroup 跟踪所有启动的 goroutine。（推荐）
		WaitGroup 是一个计数信号量，我们可以利用它来统计所有的 goroutine 是不是都完成了工作。
	*/
	var waitGroup sync.WaitGroup

	// 设置需要等待处理
	// 每个数据源的goroutine的数量
	waitGroup.Add(len(feeds))

	// 为每个数据源启动一个goroutine来查找结果
	/*
		使用关键字 for range 对 feeds 切片做迭代。
		关键字 range 可以用于迭代数组、字符串、切片、映射和通道。使用 for range 迭代切片时，每次迭代会返回两个值。
		第一个值是迭代的元素在切片里的索引位置，第二个值是元素值的一个副本。
		下划线标识符_的作用是占位符，占据了保存 range 调用返回的索引值的变量的位置。
		如果要调用的函数返回多个值，而又不需要其中的某个值，就可以使用下划线标识符将其忽略。
	*/
	for _, feed := range feeds {
		// 获取一个匹配器用于查找
		matcher, exists := matchers[feed.Type]
		//检查这个键是否存在于 map 里。如果不存在，使用默认匹配器。这样程序在不知道对应数据源的具体类型时，也可以执行，而不会中断。
		if !exists {
			/*
				查找 map 里的键时，有两个选择：要么赋值给一个变量，要么为了精确查找，赋值给两个变量。
				赋值给两个变量时第一个值和赋值给一个变量时的值一样，是 map 查找的结果值。
				如果指定了第二个值，就会返回一个布尔标志，来表示查找的键是否存在于 map 里。
				如果这个键不存在，map 会返回其值类型的零值作为返回值，如果这个键存在，map 会返回键所对应值的副本。
			*/
			matcher = matchers["default"] //通过 map 查找到一个可用于处理特定数据源类型的数据的 Matcher 值
		}

		// 启动一个goroutine来执行搜索
		/*
			一个 goroutine 是一个独立于其他函数运行的函数。使用关键字 go 启动一个 goroutine，并对这个 goroutine 做并发调度。
			用关键字 go 启动了一个匿名函数作为 goroutine。匿名函数是指没有明确声明名字的函数。
			在 for range 循环里，我们为每个数据源，以 goroutine 的方式启动了一个匿名函数。这样可以并发地独立处理每个数据源的数据。
			匿名函数也可以接受声明时指定的参数。
		*/
		go func(matcher Matcher, feed *Feed) { //匿名函数也可以接受声明时指定的参数。
			Match(matcher, feed, searchTerm, results) //Match 函数会搜索数据源的数据，并将匹配结果输出到 results 通道。
			waitGroup.Done()                          //递减 WaitGroup 的计数。
			/*
				一旦每个 goroutine 都执行调用 Match 函数和 Done 方法，程序就知道每个数据源都处理完成。
				调用 Done 方法这一行还有一个值得注意的细节：WaitGroup 的值没有作为参数传入匿名函数，但是匿名函数依旧访问到了这个值。
			*/
		}(matcher, feed) //matcher 和 feed 两个变量的值被传入匿名函数。
		/*
			在 Go 语言中，所有的变量都以值的方式传递。
			因为指针变量的值是所指向的内存地址，在函数间传递指针变量，是在传递这个地址值，所以依旧被看作以值的方式在传递。
			Go 语言支持闭包，这里就应用了闭包。实际上，在匿名函数内访问 searchTerm 和 results 变量，也是通过闭包的形式访问的。
			因为有了闭包，函数可以直接访问到那些没有作为参数传入的变量。
			匿名函数并没有拿到这些变量的副本，而是直接访问外层函数作用域中声明的这些变量本身。
			因为 matcher 和 feed 变量每次调用时值不相同，所以并没有使用闭包的方式访问这两个变量
		*/
	}

	// 启动一个goroutine来监控是否所有的工作都做完了
	go func() { //以 goroutine 的方式启动了另一个匿名函数。这个匿名函数没有输入参数，使用闭包访问了 WaitGroup 和 results 变量。
		// 等候所有任务完成
		waitGroup.Wait() //这个方法会导致 goroutine 阻塞，直到 WaitGroup 内部的计数到达 0。

		// 用关闭通道的方式，通知Display函数
		// 可以退出程序了
		close(results) //关闭 results 通道。一旦通道关闭，goroutine 就会终止，不再工作。
	}()

	// 启动函数，显示返回的结果，并且在最后一个结果显示完后返回
	Display(results) //一旦这个函数返回，程序就会终止。而之前的代码保证了所有 results 通道里的数据被处理之前，Display 函数不会返回。
}

// Register调用时，会注册一个匹配器，提供给后面的程序使用
// 将一个 Matcher 值加入到保存注册匹配器的映射中
func Register(feedType string, matcher Matcher) {
	if _, exists := matchers[feedType]; exists {
		log.Fatalln(feedType, "Matcher already registered")
	}

	log.Println("Register", feedType, "matcher")
	matchers[feedType] = matcher
}
