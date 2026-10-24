# id: C8
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: random split on iid image dataset

from sklearn.model_selection import train_test_split

imgs_tr, imgs_te, lab_tr, lab_te = train_test_split(images, labels, test_size=0.2, random_state=0)
