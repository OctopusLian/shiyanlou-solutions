#实验一：线程池实现爬虫

from queue import Queue 
from threading import Thread, Lock
import urllib.parse
import socket
import re
import time
    
seen_urls = set(['/'])
lock = Lock()
    
    
class Fetcher(Thread):
    def __init__(self, tasks):
        Thread.__init__(self)
        #tasks为任务队列
        self.tasks = tasks
        self.daemon = True
    
        self.start()
    
    def run(self):
        while True:
            url = self.tasks.get()
            print(url)
            sock = socket.socket()
            sock.connect(('localhost', 3000))
            get = 'GET {} HTTP/1.0\r\nHost: localhost\r\n\r\n'.format(url)
            sock.send(get.encode('ascii'))
            response = b''
            chunk = sock.recv(4096)
            while chunk:
                response += chunk
                chunk = sock.recv(4096)

            #解析页面上的所有链接        
            links = self.parse_links(url, response)
    
            lock.acquire()
            #得到新链接加入任务队列与seen_urls中
            for link in links.difference(seen_urls):
                self.tasks.put(link)
            seen_urls.update(links)    
            lock.release()
    
            #通知任务队列这个线程的任务完成了
            self.tasks.task_done() #在线程内调用
                
    def parse_links(self, fetched_url, response):
        if not response:
            print('error: {}'.format(fetched_url))
            return set()
        if not self._is_html(response):
            return set()
        #通过href属性找到所有链接
        urls = set(re.findall(r'''(?i)href=["']?([^\s"'<>]+)''',
                                self.body(response)))
    
        links = set()
        for url in urls:
            #可能找到的url是相对路径，这时候就需要join一下，绝对路径的话就还是会返回url
            normalized = urllib.parse.urljoin(fetched_url, url)
            #url的信息会被分段存在parts里
            parts = urllib.parse.urlparse(normalized)
            if parts.scheme not in ('', 'http', 'https'):
                continue
            host, port = urllib.parse.splitport(parts.netloc)
            if host and host.lower() not in ('localhost'):
                continue
            #有的页面会通过地址里的#frag后缀在页面内跳转，这里去掉frag的部分
            defragmented, frag = urllib.parse.urldefrag(parts.path)
            links.add(defragmented)
    
        return links

    #得到报文的html正文 
    def body(self, response):
        body = response.split(b'\r\n\r\n', 1)[1]
        return body.decode('utf-8')
    
    def _is_html(self, response):
        head, body = response.split(b'\r\n\r\n', 1)
        headers = dict(h.split(': ') for h in head.decode().split('\r\n')[1:])
        return headers.get('Content-Type', '').startswith('text/html')
    
    
class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue()
        for _ in range(num_threads):
            Fetcher(self.tasks)
    
    def add_task(self, url):
        self.tasks.put(url) #往队列添加一个url，队列满了则阻塞
    
    def wait_completion(self):
        self.tasks.join() #阻塞直到所有任务完成
    
if __name__ == '__main__':
    start = time.time()
    #开4个线程
    pool = ThreadPool(4)
    #从根地址开始抓取页面
    pool.add_task("/")
    pool.wait_completion()
    print('{} URLs fetched in {:.1f} seconds'.format(len(seen_urls),time.time() - start))