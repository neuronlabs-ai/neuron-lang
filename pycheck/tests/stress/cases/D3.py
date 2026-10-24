# id: D3
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail:
# desc: next_tok name in a trading script, used as feature

next_tok = df['close'].shift(-1)
X['f'] = next_tok
