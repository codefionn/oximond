local http = require("socket.http")
local ltn12 = require("ltn12")
http.TIMEOUT = 2
local chunks = {}
local success, code, headers, status = http.request({
   url = "http://127.0.0.1:" .. integration_http_port .. "/binary?x=1",
   sink = ltn12.sink.table(chunks),
})
assert(success == 1 and code == 200)
assert(headers["x-oximond-fixture"] == "local")
assert(status == "HTTP/1.1 200 OK")
assert(table.concat(chunks) == "loopback\0http\255\r\n")
print("socket http local binary response ok")
