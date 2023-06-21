import rospy
from std_msgs.msg import Bool
import tkinter as tk

def callback(pub):
    pub.publish(True)

def keyboard_input_publisher():
    # Initialize the ROS node
    rospy.init_node('command_node', anonymous=True)
    
    # Create a publisher with topic '/keyboard_input' and message type 'Bool'
    pub = rospy.Publisher('/start_detection_command', Bool, queue_size=1)
    
    # create root window
    root = tk.Tk()
    
    # root window title and dimension
    root.title("Detect Objects")
    # Set geometry (widthxheight)
    root.geometry('350x200')
    

    button = tk.Button(root, text='Detect Objects', width=25, command=lambda: callback(pub))
    button.grid()

    #button to kill the program
    buttonEnd = tk.Button(root, text='End Detection', width=25, command=root.destroy)
    buttonEnd.grid(column=1, row=0)


    #when enter is pressed, call the function callback with the argument 'pub'
    root.bind('<Return>', lambda event: callback(pub))

    # all widgets will be here
    # Execute Tkinter
    root.mainloop()
    



if __name__ == '__main__':
    try:
        keyboard_input_publisher()
    except rospy.ROSInterruptException:
        pass
