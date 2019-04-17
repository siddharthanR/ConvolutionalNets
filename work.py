from datetime import datetime as dt
s = dt.now().time()
e = dt.now().time()

print(s.hour - e.hour)
print(s.minute - e.minute)
print(s.second - e.second)