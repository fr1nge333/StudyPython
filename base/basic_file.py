'''
文件读取
'''
with open('../wx_auto_msg/msg.txt',encoding='utf8',mode='r') as msg_file:
    for msg in msg_file:
        print(msg)
msg_file.close()

