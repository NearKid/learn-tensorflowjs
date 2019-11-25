const fs = require('fs');
const mime = require('mime');
const http = require('http');
const url = require('url');
const path = require('path');

const server = http.createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "X-Requested-With");
  res.setHeader("Access-Control-Allow-Methods", "PUT,POST,GET,DELETE,OPTIONS");
  res.setHeader("X-Powered-By", ' 3.2.1')
  res.setHeader("Content-Type", "application/json;charset=utf-8");

  const reqUrl = decodeURIComponent(req.url);
  const obj = url.parse(reqUrl);
  const pathname = obj.pathname;

  const realPath = path.join(__dirname, '/public', pathname);
  fs.stat(realPath, (err, stats) => {
    if (err || stats.isDirectory()) {
      res.writeHead(404, 'not found', {
        'Content-Type': 'text/plain'
      });
      res.write(`the request ${pathname} is not found`);
      res.end();
    } else {
      let ext = path.extname(realPath).slice(1); // 获取文件拓展名
      contentType = mime.getType(ext) || 'text/plain';
      res.setHeader('content-type', contentType);
      let raw = fs.createReadStream(realPath);
      res.writeHead(200, 'ok');
      raw.pipe(res);
    }
  });
});

server.listen(8000);
console.log('server start at http://localhost:8000')