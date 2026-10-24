# id: I2_iid_split_images
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: random split on image dataset

from sklearn.model_selection import train_test_split

tr, te = train_test_split(images, labels, test_size=0.2, random_state=0)
