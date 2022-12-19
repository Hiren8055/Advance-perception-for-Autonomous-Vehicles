
def transform(myBox):
    '''
    Transformation of topview coords wrt to 1000,0 as origin. Translation done wrt car pose (208, 416)
    :x_axis_constant: Translate in x (1000 - 208)
    :y_axis_constant: Translate in y (1000 - 416)
    
    :returns: Translated obstacle list   
    '''
    transformed = []
    y_axis_constant = 416
    x_axis_constant = 792
    for i in myBox:
        y = y_axis_constant - i[1]
        x = i[0] + x_axis_constant
        
        if y <= 410:
            transformed.append(tuple((x, y)))
            
    return transformed


def ground_truth_top_view(myBox):
    '''
    Calculate the ground truth topview
    : car_position: 1000,0
    : ratio: multiplication factor for getting on_ground values (ground:map ratio)
    : axis: lines in top view to change the ratio
    : returns corrected top view list
    '''
    
    
    ######## X Axis scaling constants
    # x_ratios = [0, 2, 2.67, 3.33, 4]
    # lines = [0, 90, 130, 180, 300]
    # ground_x = [0, 180, 286, 452, 932]
    x_ratio = 1.6
    car_x = 1000
    
    ######## Y Axis scaling constants
    y_ratios = [2, 2.16 ,2.28, 2.5, 2.67, 2.96]
    # ground_axis_lines values are calculated by calculate_ground_axis_lines function
    ground_axis_lines = [0, 200, 416.0, 666.8, 841.8, 895.2, 942.56] 
    map_axis_lines = [0, 100, 200, 310, 380, 400, 416]
    error = 10
    
    correct = []
    count = 0
    
    def x_correction(x):
        # diff = abs(car_x-x)
        # for i in range(1,len(x_ratios)):
        #     if diff <= lines[i]:
        #         if x > car_x:
        #             temp =  (x-(car_x+lines[i-1]))*x_ratios[i]
        #             x = car_x + ground_x[i-1] + temp
        #         else:
        #             temp = ((car_x - lines[i-1]) - x)*x_ratios[i]
        #             x = car_x - ground_x[i-1] - temp
        #         return x, True
        # print("Didn't find X")
        # return x, False

        diff = abs(car_x-x)

        if x > car_x:
            x = car_x + (diff * x_ratio)
        else:
            x = car_x - (diff * x_ratio)

        return x
    

    for point in myBox:

        x = point[0]
        y = 0
        
        for j in range(1,len(map_axis_lines)):
            if point[1] > map_axis_lines[j-1] and point[1] <= (map_axis_lines[j]+error):
                temp = point[1] - map_axis_lines[j-1]
                y = ground_axis_lines[j-1] + (temp * y_ratios[j-1])

                # Car as point object should be considered from front of car
                # Can change is to y -= 80
                y -= 40 # removing and extra tile on ground

                x = x_correction(x)
                x = round(x, 2)
                y = round(y, 2)

                correct.append(tuple((x, y)))

                break
    
    return correct


def get_corrected_top_view(top_view):
    
    transformed = transform(top_view)
    # print("transformed",transformed)
    top_view = ground_truth_top_view(transformed)
    
    return top_view
    #return transformed
def calculate_ground_axis_lines(map_axis_lines,y_ratios):    
    new_ground_line_arr = [0]    
    for i in range(1,len(map_axis_lines)):
        diff = map_axis_lines[i] - map_axis_lines[i-1]
        new_ground_line = new_ground_line_arr[i-1]+y_ratios[i-1]*diff
        print("y_ratios :", y_ratios[i-1])
        print("ground_axis_lines :", new_ground_line_arr[i-1])
        print("new_ground_line :", new_ground_line)
        new_ground_line =round(new_ground_line,3)      
        new_ground_line_arr.append(new_ground_line)
    print(new_ground_line_arr)

def main():
    opt = int(input(("1 for transforamtion \n2 for calculating ground_axis_lines \n")))
    if opt == 1:
        # myBox = [(144, 324), (167, 304), (187, 283), (206, 264), (225, 245), (244, 227),(261, 211), (133, 171), (152, 155), (170, 139), (218, 104), (233, 93),(278, 196), (295, 177), (187, 130), (203, 117), (247, 81), (262, 72)]
        # myBox = [(381,79),(404,-32),(426,-183),(458,-376),(505,-564),(596,-758),(694,-953),(786,-1118),(43,104),(36,-32),(44,-174),(272,-322),(66,-322),(366,-470),(93,-470),(129,-630),(158,-817),(221,-953),(288,-1118)]
        # myBox = [(-66,-231),(-82,-376),(344,155),(335,38),(126,-5),(321,-68),(155,-75),(28,-110),(305,-201),(289,-390),(273,-544),(263,-630),(241,-787)]
        myBox = [(83,208),(91,100),(70,-38),(32,-148),(0,-241),(300,-62),(228,-148),(426,-157),(404,-285),(371,-405),(326,-525),(266,-607),(212,-703)]
        ### Transform map -> ground
        transformed = transform(myBox)
        print("transformed: ", transformed)
        # top_view = ground_truth_top_view(transformed)
    if opt == 2:
        y_ratios = [2, 2.16 ,2.28, 2.5, 2.67, 2.96]
        map_axis_lines = [0, 100, 200, 310, 380, 400, 416]
        calculate_ground_axis_lines(map_axis_lines,y_ratios)

if __name__ == "__main__":
    main()

'''
https://www.desmos.com/calculator/qckmssg6st
'''