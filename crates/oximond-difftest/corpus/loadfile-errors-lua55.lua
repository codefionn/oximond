-- oximond-difftest repro profile=lua55
-- loadfile and dofile report an unopenable file as C's strerror does.
print(pcall(dofile, "/nonexistent/file.lua"))
print(loadfile("/nonexistent/file.lua"))
print(pcall(loadfile, "/nonexistent/dir/"))
print(loadfile("/tmp"))
print(pcall(dofile, "/tmp"))
print(pcall(dofile, ""))
print(loadfile(""))
