import rclpy
from std_msgs.msg import Bool
import tkinter as tk

def callback(pub):
    msg = Bool()
    msg.data = True
    pub.publish(msg)

def keyboard_input_publisher():
    # Initialize the ROS node
    rclpy.init()
    node = rclpy.create_node('command_node') #, anonymous=True)
    
    # Create a publisher with topic '/keyboard_input' and message type 'Bool'
    pub = node.create_publisher(Bool, '/start_detection_command', 1)
    
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
    except rclpy.ROSInterruptException:
        pass
