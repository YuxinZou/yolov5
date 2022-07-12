import utils.autoanchor as autoAC
 
# 对数据集重新计算 anchors
new_anchors = autoAC.kmean_anchors('./data/qianhai_clean_add0616_public_bdd100k.yaml', 9, 1280, 8.0, 1000, True)
print(new_anchors)
