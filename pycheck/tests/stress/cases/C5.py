# id: C5
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: shift(-1) only in string and comment

msg = "never use shift(-1) in features"  # .shift(-1) is a leak
