package gocache

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

type Item struct {
	Object     interface{} // 真正的数据项,用于存储任意类型的数据对象
	Expiration int64       // 生存时间,存储了该数据项的过期时间
}

// 判断数据项是否已经过期
func (item Item) Expired() bool {
	if item.Expiration == 0 {
		return false
	}
	return time.Now().UnixNano() > item.Expiration //注意，数据项的过期时间，是 Unix 时间戳，单位是纳秒
}

/*
怎么样判断数据项有没有过期呢？
其实非常简单。我们在每一个数据项中，记录数据项的过期时间，然后缓存系统将定期检查每一项数据项，
如果发现数据项的过期时间小于当前时间，则将数据项从缓存系统中删除。
这里我们将借助 [time](https://godoc.org/time) 模块来实现周期任务。
*/

const (
	// 没有过期时间标志
	NoExpiration time.Duration = -1 //数据项永远不过期

	// 默认的过期时间
	DefaultExpiration time.Duration = 0 //标记数据项应该拥有一个默认过期时间
)

type Cache struct {
	defaultExpiration time.Duration
	items             map[string]Item // 缓存数据项存储在 map 中
	mu                sync.RWMutex    // 读写锁
	gcInterval        time.Duration   // 过期数据项清理周期
	stopGc            chan bool
}

// 过期缓存数据项清理
/*
该方通过`time.Ticker` 定期执行 `DeleteExpired()` 方法，从而清理过期的数据项。
通过 `time.NewTicker()` 方法创建的 ticker, 会通过指定的`c.Interval` 间隔时间，周期性的从 `ticker.C` 管道中发送数据过来，
我们可以根据这一特性周期性的执行`DeleteExpired()` 方法。
同时为使 `gcLoop()`函数能正常结束，我们通过监听`c.stopGc`管道，如果有数据从该管道中发送过来，我们就停止`gcLoop()` 的运行。
*/
func (c *Cache) gcLoop() {
	ticker := time.NewTicker(c.gcInterval)
	for {
		select {
		case <-ticker.C:
			c.DeleteExpired()
		case <-c.stopGc:
			ticker.Stop()
			return
		}
	}
}

// 删除缓存数据项
func (c *Cache) delete(k string) {
	delete(c.items, k)
}

// 删除过期数据项
/*
只需要遍历所有数据项，删除过期数据即可
*/
func (c *Cache) DeleteExpired() {
	now := time.Now().UnixNano()
	c.mu.Lock()
	defer c.mu.Unlock()

	for k, v := range c.items {
		if v.Expiration > 0 && now > v.Expiration {
			c.delete(k)
		}
	}
}

// 设置缓存数据项，如果数据项存在则覆盖
func (c *Cache) Set(k string, v interface{}, d time.Duration) {
	var e int64
	if d == DefaultExpiration {
		d = c.defaultExpiration
	}
	if d > 0 {
		e = time.Now().Add(d).UnixNano()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[k] = Item{
		Object:     v,
		Expiration: e,
	}
}

// 设置数据项, 没有锁操作
func (c *Cache) set(k string, v interface{}, d time.Duration) {
	var e int64
	if d == DefaultExpiration {
		d = c.defaultExpiration
	}
	if d > 0 {
		e = time.Now().Add(d).UnixNano()
	}
	c.items[k] = Item{
		Object:     v,
		Expiration: e,
	}
}

// 获取数据项，如果找到数据项，还需要判断数据项是否已经过期
func (c *Cache) get(k string) (interface{}, bool) {
	item, found := c.items[k]
	if !found {
		return nil, false
	}
	if item.Expired() {
		return nil, false
	}
	return item.Object, true
}

// 添加数据项，如果数据项已经存在，则返回错误
/*
Set 和 Add：能避免缓存被错误的覆盖
*/
func (c *Cache) Add(k string, v interface{}, d time.Duration) error {
	c.mu.Lock()
	_, found := c.get(k)
	if found {
		c.mu.Unlock()
		return fmt.Errorf("Item %s already exists", k)
	}
	c.set(k, v, d)
	c.mu.Unlock()
	return nil
}

// 获取数据项
func (c *Cache) Get(k string) (interface{}, bool) {
	c.mu.RLock()
	item, found := c.items[k]
	if !found {
		c.mu.RUnlock()
		return nil, false
	}
	if item.Expired() {
		return nil, false
	}
	c.mu.RUnlock()
	return item.Object, true
}

// 替换一个存在的数据项
func (c *Cache) Replace(k string, v interface{}, d time.Duration) error {
	c.mu.Lock()
	_, found := c.get(k)
	if !found {
		c.mu.Unlock()
		return fmt.Errorf("Item %s doesn't exist", k)
	}
	c.set(k, v, d)
	c.mu.Unlock()
	return nil
}

// 删除一个数据项
func (c *Cache) Delete(k string) {
	c.mu.Lock()
	c.delete(k)
	c.mu.Unlock()
}

// 将缓存数据项写入到 io.Writer 中
/*
通过gob模块将二进制缓存数据转码写入到实现了io.Writer接口的对象中
*/
func (c *Cache) Save(w io.Writer) (err error) {
	enc := gob.NewEncoder(w)
	defer func() {
		if x := recover(); x != nil {
			err = fmt.Errorf("Error registering item types with Gob library")
		}
	}()
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, v := range c.items {
		gob.Register(v.Object)
	}
	err = enc.Encode(&c.items)
	return
}

// 保存数据项到文件中
func (c *Cache) SaveToFile(file string) error {
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	if err = c.Save(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// 从 io.Reader 中读取数据项
/*
从io.Reader中读取二进制数据，然后通过gob模块将数据进行反序列化
*/
func (c *Cache) Load(r io.Reader) error {
	dec := gob.NewDecoder(r)
	items := map[string]Item{}
	err := dec.Decode(&items)
	if err == nil {
		c.mu.Lock()
		defer c.mu.Unlock()
		for k, v := range items {
			ov, found := c.items[k]
			if !found || ov.Expired() {
				c.items[k] = v
			}
		}
	}
	return err
}

// 从文件中加载缓存数据项
func (c *Cache) LoadFile(file string) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	if err = c.Load(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// 返回缓存数据项的数量
func (c *Cache) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// 清空缓存
func (c *Cache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = map[string]Item{}
}

// 停止过期缓存清理
func (c *Cache) StopGc() {
	c.stopGc <- true
}

// 创建一个新的缓存系统
func NewCache(defaultExpiration, gcInterval time.Duration) *Cache {
	c := &Cache{
		defaultExpiration: defaultExpiration,
		gcInterval:        gcInterval,
		items:             map[string]Item{},
		stopGc:            make(chan bool),
	}
	// 开始启动过期清理 goroutine
	go c.gcLoop()
	return c
}
