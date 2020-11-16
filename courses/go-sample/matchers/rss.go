package matchers

import (
	"encoding/xml"
	"errors"
	"fmt"
	"log"
	"net/http"
	"regexp"

	"../search"
)

type (
	// item根据item字段的标签，将定义的字段与rss文档的字段关联起来
	item struct {
		XMLName     xml.Name `xml:"item"`
		PubDate     string   `xml:"pubDate"`
		Title       string   `xml:"title"`
		Description string   `xml:"description"`
		Link        string   `xml:"link"`
		GUID        string   `xml:"guid"`
		GeoRssPoint string   `xml:"georss:point"`
	}

	// image根据image字段的标签，将定义的字段与rss文档的字段关联起来
	image struct {
		XMLName xml.Name `xml:"image"`
		URL     string   `xml:"url"`
		Title   string   `xml:"title"`
		Link    string   `xml:"link"`
	}

	// channel根据channel字段的标签，将定义的字段与rss文档的字段关联起来
	channel struct {
		XMLName        xml.Name `xml:"channel"`
		Title          string   `xml:"title"`
		Description    string   `xml:"description"`
		Link           string   `xml:"link"`
		PubDate        string   `xml:"pubDate"`
		LastBuildDate  string   `xml:"lastBuildDate"`
		TTL            string   `xml:"ttl"`
		Language       string   `xml:"language"`
		ManagingEditor string   `xml:"managingEditor"`
		WebMaster      string   `xml:"webMaster"`
		Image          image    `xml:"image"`
		Item           []item   `xml:"item"`
	}

	// rssDocument定义了与rss文档关联的字段
	rssDocument struct {
		XMLName xml.Name `xml:"rss"`
		Channel channel  `xml:"channel"`
	}
)

// rssMatcher 实现了Matcher接口
/*
这个声明与 defaultMatcher 类型的声明很像。因为不需要维护任何状态，所以我们使用了一个空结构来实现 Matcher 接口。
*/
type rssMatcher struct{}

// init 将匹配器注册到程序里
func init() {
	var matcher rssMatcher
	search.Register("rss", matcher)
}

// Search在文档中查找特定的搜索项
func (m rssMatcher) Search(feed *search.Feed, searchTerm string) ([]*search.Result, error) {
	/*
		使用关键字 var 声明了一个值为 nil 的切片，切片每一项都是指向 Result 类型值的指针。
	*/
	var results []*search.Result //这个变量用于保存并返回找到的结果。

	log.Printf("Search Feed Type[%s] Site[%s] For URI[%s]\n", feed.Type, feed.Name, feed.URI)

	// 获取要搜索的数据
	document, err := m.retrieve(feed)
	if err != nil {
		return nil, err
	}

	/*
		调用 retrieve 方法返回了一个指向 rssDocument 类型值的指针以及一个错误值。之后，像已经多次看过的代码一样，检查错误值，
		如果真的是一个错误，直接返回。如果没有错误发生，之后会依次检查得到的 RSS 文档的每一项的标题和描述，如果与搜索项匹配，
		就将其作为结果保存
	*/
	for _, channelItem := range document.Channel.Item {
		// 检查标题部分是否包含搜索项
		matched, err := regexp.MatchString(searchTerm, channelItem.Title) //对 channelItem 值里的 Title 字段进行搜索，查找是否有匹配的搜索项
		if err != nil {
			return nil, err
		}

		// 如果找到匹配的项，将其作为结果保存
		if matched {
			results = append(results, &search.Result{
				Field:   "Title",
				Content: channelItem.Title,
			})
		}

		// 检查描述部分是否包含搜索项
		matched, err = regexp.MatchString(searchTerm, channelItem.Description)
		if err != nil {
			return nil, err
		}

		// 如果找到匹配的项，将其作为结果保存
		if matched {
			/*
				append 这个内置函数会根据切片需要，决定是否要增加切片的长度和容量。
				这个函数的第一个参数是希望追加到的切片，第二个参数是要追加的值。

				例子：追加到切片的值是一个指向 Result 类型值的指针。
				这个值直接使用字面声明的方式，初始化为 Result 类型的值。
				之后使用取地址运算符（&），获得这个新值的地址。最终将这个指针存入了切片。
			*/
			results = append(results, &search.Result{ //将搜索结果加入到 results 切片里
				Field:   "Description",
				Content: channelItem.Description,
			})
		}
	}

	return results, nil //返回了 results 作为函数调用的结果
}

// retrieve发送HTTP Get请求获取rss数据源并解码
/*
方法 retrieve 并没有对外暴露，其执行的逻辑是从 RSS 数据源的链接拉取 RSS 文档。
*/
func (m rssMatcher) retrieve(feed *search.Feed) (*rssDocument, error) {
	if feed.URI == "" {
		return nil, errors.New("No rss feed uri provided")
	}

	// 从网络获得rss数据源文档
	/*
		使用 http 包，Go 语言可以很容易地进行网络请求。当 Get 方法返回后，我们可以得到一个指向 Response 类型值的指针。
		之后会监测网络请求是否出错，并安排函数返回时调用 Close 方法。
	*/
	resp, err := http.Get(feed.URI) //调用了 http 包的 Get 方法
	if err != nil {
		return nil, err
	}

	// 一旦从函数返回，关闭返回的响应链接
	defer resp.Body.Close()

	// 检查状态码是不是200，这样就能知道是不是收到了正确的响应
	/*
		任何不是 200 的请求都需要作为错误处理。如果响应值不是 200，我们使用 fmt 包里的 Errorf 函数返回一个自定义的错误。
	*/
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP Response Error %d\n", resp.StatusCode)
	}

	// 将rss数据源文档解码到我们定义的结构类型里,不需要检查错误，调用者会做这件事
	var document rssDocument
	/*
		NewDecoder函数：
		这个函数会返回一个指向 Decoder 值的指针。之后调用这个指针的 Decode 方法，传入 rssDocument 类型的局部变量 document 的地址。
		最后返回这个局部变量的地址和 Decode 方法调用返回的错误值。
	*/
	err = xml.NewDecoder(resp.Body).Decode(&document) //使用 xml 包并调用了同样叫作 NewDecoder 的函数。
	return &document, err
}
