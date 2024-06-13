#imports
import cv2
import mediapipe as mp
import click
import sqlite3
from sqlite3 import Error
import os
from collections import deque
import csv
import time

##########################################################
#global variable setup
mp_holistic = mp.solutions.holistic

#Holistic class setup
holistic = mp_holistic.Holistic(
    static_image_mode = True,
    model_complexity = 2,
    min_detection_confidence = 0.75,
)
##########################################################

def most_frequent(List):
    """
    returns the most frequent element from the input list
    """
    counter = 0
    word = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            word = i
 
    return word


def getBoundingCoord(landmark, height, width):
    """
    Returns bounding box coordinates in 4 tuple
    i.e. (min x, min y, max x, max y)

    returns None if both hands are below waist
    """

    # full landmarks generated regardless of the what part of the person is given
    if ((landmark[23].y * height) > (landmark[15].y * height)) or ((landmark[24].y * width) > (landmark[16].y * width)):

        # if any of the hands are above the waistline

        #separate out nose, left hand, and right hand landmarks
        collection = [landmark[0]] + landmark[15:23]

        # get min x. Using sort for O(n log n).
        # note: min() has O(n)
        x_ascd = sorted(collection, key = lambda x: x.x)

        # get min y
        y_ascd = sorted(collection, key = lambda y: y.y)

        # bounding coordinates (min x, min y, max x, max y)
        return (x_ascd[0].x, y_ascd[0].y, x_ascd[-1].x, y_ascd[-1].y)
    
    return None


def getHandBoundingCoord(landmarks):
    """ 
    Returns four corner points of bounding box for the hand landmarks
    Does not take account for the original image size apply universally across all
    image sizes
    """
  
    #find max and min
    x_max = max(landmarks, key=lambda x: x.x).x
    y_max = max(landmarks, key=lambda y: y.y).y
    x_min = min(landmarks, key=lambda x: x.x).x
    y_min = min(landmarks, key=lambda y: y.y).y

    return (x_max, y_max, x_min, y_min)


def getPrimaryQuad(bounding_coord):
    """
    Get primary quad (16 quadrants) boundary dictionary
    {
        (columns: A, B, C, D, E), 
        (rows: 0, 1, 2, 3, 4)
    }

    Pulled out as a separate function for easier human reading
    """

    x_min, y_min, x_max, y_max = bounding_coord

    x_q = (x_max - x_min) / 4
    y_q = (y_max - y_min) / 4
 
    return {
        "A": x_min,
        "B": x_min + x_q,
        "C": x_min + (x_q * 2),
        "D": x_max - x_q,
        "E": x_max,
        "0": y_max,
        "1": y_max - y_q,
        "2": y_max - (y_q * 2),
        "3": y_min + y_q,
        "4": y_min
    }


def getCoordinatesIntoQuads(landmarks, primary_quad):
    """
    places each landmark into quads within the bounding box
    Returns string of all points
    """

    # Just nose and hands
    collection = [landmarks[0]] + landmarks[15:23]

    coord_code = ""
    
    for i, landmark in enumerate(collection):
        if (primary_quad["A"] <= landmark.x <= primary_quad["E"]) and (primary_quad["4"] <= landmark.y <= primary_quad["0"]):
            #get x position
            if landmark.x >= primary_quad["C"]:
                if landmark.x >= primary_quad["D"]:
                    coord_code += "D"
                else:
                    coord_code += "C"
            else:
                if landmark.x < primary_quad["B"]:
                    coord_code += "A"
                else:
                    coord_code += "B"
            
            #get y position
            if landmark.y >= primary_quad["2"]:
                if landmark.y >= primary_quad["1"]:
                    coord_code += "0"
                else:
                    coord_code += "1"
            else:
                if landmark.y >= primary_quad["3"]:
                    coord_code += "2"
                else:
                    coord_code += "3"
            
            # add index
            coord_code += f"{i}"

    return coord_code


def getHandCoordinatesIntoQuads(landmarks, primary_quad):
    """
    Helper function. Places each landmarks into quadrants
    """

    coord_code = ""
    
    for i, landmark in enumerate(landmarks):

        #get x position
        if landmark.x >= primary_quad["C"]:
            if landmark.x >= primary_quad["D"]:
                coord_code += "D"
            else:
                coord_code += "C"
        else:
            if landmark.x < primary_quad["B"]:
                coord_code += "A"
            else:
                coord_code += "B"
        
        #get y position
        if landmark.y >= primary_quad["2"]:
            if landmark.y >= primary_quad["1"]:
                coord_code += "0"
            else:
                coord_code += "1"
        else:
            if landmark.y >= primary_quad["3"]:
                coord_code += "2"
            else:
                coord_code += "3"
        
        # add index
        coord_code += f"{i}"

    return coord_code



def getQuadString(result, image):
    """ wrapper function """

    height, width, _ = image.shape # get height and width of the image. discard z length

    bounding_coord = False #for when image does not contain a human figure

    #get bounding coordinates
    if result.pose_landmarks: #check if landmarks are present
        bounding_coord = getBoundingCoord(result.pose_landmarks.landmark, height, width)

    if bounding_coord:
        primary_quad = getPrimaryQuad(bounding_coord)

        return getCoordinatesIntoQuads(result.pose_landmarks.landmark, primary_quad)

    return False


def create_connection(db_file):
    """
    Creates connection to the SQlite db (local)
    """
    conn = None

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)



def img_translate(image, prev, c):
    """ Translates given sign. Return prev (previous word) if no sign detected """
    if type(image) == str:
        image = cv2.imread(image)

    result = holistic.process(image)

    quadstring = getQuadString(result, image)

    if quadstring:
        left_hand, right_hand = getHandString(result)
        print("left: %s" %left_hand)
        print("right: %s" %right_hand)

        if left_hand or right_hand:
            c.execute(f"SELECT label FROM Sign WHERE body='{quadstring}' AND \
                        lefthand='{left_hand}' AND righthand='{right_hand}'")
            r = c.fetchone()[0]
            if r == None:
                return prev
            else:
                return r

    return prev


def display(frame, output, vid_state):
    """
    displays output onto the screen
    """
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(frame, output, (50,50), font, 1, (0,255,255), 2, cv2.LINE_4)
    cv2.imshow(vid_state, frame)


def cam_reader():
    """ Processes cam live input """
    video_path = -1
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit")

    
    output = "starting"

    #default queue.
    common = deque(["starting", "starting", "starting", "starting", "starting"])

    while True:
        ret, frame = video_capture.read()

        #check if frame is read properly
        if not ret:
            print("Cannot receive frame. Exiting...")
            break

        # Translate the sign. Return previous if none detected
        common.popleft()
        common.append(img_translate(frame, output))

        output = most_frequent(common)
        print(output)

        display(frame, output, "output")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()



def img_reader(source):
    """ Processes img input """

    img = source

    if type(source) is str:
        img = cv2.imread(source)

    output = img_translate(img, "No sign detected")

    display(img, output, "output")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()



def vid_reader(source):
    """ Processes mp4 video input """

    video_capture = cv2.VideoCapture(source)

    if not video_capture.isOpened():
        print("Cannot open video file")
        exit()

    print("Press 'q' to quit")

    output = "starting"

    common = deque(["starting", "starting", "starting"])
    conn = create_connection("db/pythonsqlite.db")
    c = conn.cursor()

    while True:
        ret, frame = video_capture.read()

        #check if frame is read properly
        if not ret:
            print("Cannot receive frame. Exiting...")
            break

        # Translate the sign. Return previous if none detected
        common.popleft()
        common.append(img_translate(frame, output, c))

        output = most_frequent(common)

        display(frame, output, "output")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    conn.close()
    cv2.destroyAllWindows()


def read(source):
    """ Reads input and displays text translation on the display"""
    #Check if correct source
    format_list = {".jpg": 1, "jpeg": 1, ".png": 1, ".mp4": 2}

    if source == "cam":
        cam_reader()

    format_type = format_list.get(source[-4:], 4)

    if format_type == 1:
        img_reader(source)

    elif format_type == 2:
        vid_reader(source)

    else:
        print("Unsupported source format. Please check again") 


def getHandString(result):
    """Returns coord string for left and right hand"""
    left = ""
    right = ""

    #left hand
    if (result.left_hand_landmarks):
        l_bound = getHandBoundingCoord(result.left_hand_landmarks.landmark)
        l_prime_quad = getPrimaryQuad(l_bound)
        left = getHandCoordinatesIntoQuads(result.left_hand_landmarks.landmark, l_prime_quad)
  
    if (result.right_hand_landmarks):
        r_bound = getHandBoundingCoord(result.right_hand_landmarks.landmark)
        r_prime_quad = getPrimaryQuad(r_bound)
        right = getHandCoordinatesIntoQuads(result.right_hand_landmarks.landmark, r_prime_quad)

    return left, right


def batch_learner(source, label):
    """
    CLI tool function to learn batches of video frames at once with same label.
    mp4 input: splits video into frames and feeds each frame through the learn() function
    with single label
    directory input: feeds all image files found withint the source directory through 
    the learn() function
    """
    if source[-4:] == ".mp4":
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Could not open the file.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Exiting...")
                break

            learn(frame, label)
        
    else:
        for img in os.listdir(source):
            file = os.path.join(source, img)
            if os.path.isfile(file):
                learn(file, label)
    
        print("Finished learning")

    

def learn(source, label=None):
    """Insert new sign's string coordinates and label into the db"""

    # load the target image
    if type(source) == str:
        image = cv2.imread(source) #returns numpy array (x, y, z)
    else:
        image = source

    # get results as numpy array
    result = holistic.process(image)

    quadString = getQuadString(result, image)
    mute = True

    if quadString:
        left_hand, right_hand = getHandString(result)

        #Ask user to label the learned image. Converts entry into a string
        if label == None:
            label = str(input("Processing successful. Please label the sign: "))
            mute = False

        try:
            conn = create_connection("db/pythonsqlite.db")
            c = conn.cursor()
            c.execute(f"SELECT label FROM Sign WHERE body='{quadString}' AND \
                      lefthand='{left_hand}' AND righthand='{right_hand}'")
            if c.fetchone() == None:
                c.execute(f"INSERT INTO Sign (body, lefthand, righthand, label) VALUES ('{quadString}', '{left_hand}', '{right_hand}', '{label}');")
                conn.commit()
                print("Success")
            else:
                print("Already in the database")
            conn.close()
        except Error as e:
            print(e)

    
    else:
        if not mute:
            print("No sign detected from the image. Please check the source image.")
    



def fresh_db():
    """Wipe and set up db for clean slate state"""
    conn = create_connection("db/pythonsqlite.db")
    c = conn.cursor()

    c.execute(f"DROP TABLE Sign;")

    #remake each table
    #at the moment only one table, ha
    table_str = "CREATE TABLE Sign ( \
        Signid integer PRIMARY KEY AUTOINCREMENT, \
        body text, \
        lefthand text, \
        righthand text, \
        label text \
    );"

    c.execute(table_str)

    c.close()

            
def analyse(source):
    """
    Records translation of each frame. Records None if no sign detected
    note: Headless.
    """
    start = time.time()
    filename = f"analysis.csv"

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        video_capture = cv2.VideoCapture(source)

        if not video_capture.isOpened():
            print("Cannot open the source file. Mp4 file formats only")
            exit()

        frame_num = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Cannot receive frame. Exiting.")
                break
            csvwriter.writerow([frame_num, img_translate(frame, "None")])
            frame_num += 1

            if (frame_num % 100 == 0):
                print("Processed frame: %d" %frame_num) 

            if frame_num >= 10000:
                break

        video_capture.release()
        end = time.time()
        csvwriter.writerow(["Time", end - start])
    csvfile.close()
    print("Finished analysis.")



@click.command()
@click.option('--init', default='read', help='Initiate in "read" or "learn" mode')
@click.option('--source', required=True, type=str, help='Source to process. "cam" for live camera, "<filename>" for file')
@click.option('--fresh', default=False, help='Use this option to wipe db and start with blank state')
@click.option('--batch', default=False, help='batch upload of a single sign for learning' )
@click.option('--analysis', default=False)
def main(init, source, fresh, batch, analysis):
    if fresh == True:
        fresh_db()

    if init == 'read':
        if analysis:
            analyse(source) 
        else:
            read(source)

    else:
        if source != "cam":
            try:
                if batch:
                    label = input("Batch learning. Label? ")
                    batch_learner(source, label)

                else:
                    learn(source)
            except Error as e:
                print(e)
        else:
            print("Live camera feed cannot be used for learning.")


# # Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	main()
    