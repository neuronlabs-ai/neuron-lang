fn main():
  let s = Stream(4)
  s.push(10.0)
  let illegal = s.lead(1)
  print(illegal)
