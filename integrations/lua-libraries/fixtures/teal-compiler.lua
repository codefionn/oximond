local tl = require("tl")
assert(tl.version() == "0.24.8")
local good = [[
local record Point
   x: number
   y: number
end
local function sum<T>(values: {T}, fold: function(T, T): T): T
   local result = values[1]
   for i = 2, #values do
      result = fold(result, values[i])
   end
   return result
end
local p: Point = {x = 20, y = 22}
local label: string = "\u{03bb}"
assert(#label == 2)
local answer = sum({p.x, p.y}, function(a: number, b: number): number
   return a + b
end)
return answer
]]
local env = assert(tl.init_env(false, "off", "5.1"))
local result = assert(tl.process_string(good, false, env, "fixture.tl"))
assert(#result.syntax_errors == 0 and #result.type_errors == 0)
local code = assert(tl.pretty_print_ast(result.ast, "5.1"))
local generated = assert((loadstring or load)(code, "@fixture.lua"))
assert(generated() == 42)
print("generated", code)
local bad = "local answer: number = 'wrong'\nreturn answer\n"
local failed = assert(tl.process_string(bad, false, assert(tl.init_env(false, "off", "5.1")), "bad.tl"))
assert(#failed.syntax_errors == 0 and #failed.type_errors > 0)
for _, diagnostic in ipairs(failed.type_errors) do
   print("type", diagnostic.y, diagnostic.x, diagnostic.msg)
end
local malformed = assert(tl.process_string("local x: = 1", false,
   assert(tl.init_env(false, "off", "5.1")), "syntax.tl"))
assert(#malformed.syntax_errors > 0)
for _, diagnostic in ipairs(malformed.syntax_errors) do
   print("syntax", diagnostic.y, diagnostic.x, diagnostic.msg)
end
print("teal compiler generated execution diagnostics ok")
