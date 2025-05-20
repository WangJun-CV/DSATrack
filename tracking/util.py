def calculate_iou(box1,box2):##x,y,w,h
    x1,y1,w1,h1=box1
    x2,y2,w2,h2=box2
    intersection_x=max(0,min(x1+w1,x2+w2)-max(x1,x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area=intersection_x*intersection_y
    union_area=w1*h1+w2*h2-intersection_area
    iou = intersection_area/union_area if union_area>0 else 0
    return iou




