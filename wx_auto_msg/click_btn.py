import sys
import time

import pyautogui
import pyperclip
import pyscreeze

import image_identify.img_match as img_match


def img_check(template_address):
    '''
    截取当前屏幕画面，并识别是否包含模板图片
    :param template_address: 模版图地址
    :return: 如果包含模板图片，则返回模板图片中心坐标;否则返回(0,0)
    '''
    x = -1
    y = -1
    # 获取屏幕截图
    pyscreeze.screenshot("../img/sc.png")
    # 识别
    x, y = img_match.img_match_by_FLANN("../img/sc.png", template_address)
    if x == 0 and y == 0:
        x, y = img_match.img_match_by_RGB2gray("../img/sc.png", template_address)

    return x, y


def search_friends(x, y, name):
    '''
    通过名字搜索发信人，并打开对话框
    :param x:
    :param y:
    :param name:
    :return:
    '''
    pyautogui.moveTo(x, y, 0.5)
    pyautogui.click()
    pyperclip.copy(name)
    pyautogui.hotkey('ctrlleft', 'v')
    time.sleep(0.5)
    pyautogui.press('enter')
    print('press enter1')
    # time.sleep(0.5)
    # pyautogui.press('enter')
    # print('press enter2')


def type_msg(message):
    '''
    清除当前输入框的内容并输入新的内容
    :param message: 需要输入的内容
    :return: null
    '''
    pyautogui.hotkey('ctrlleft', 'a')
    pyautogui.press('backspace')
    pyperclip.copy(message)
    pyautogui.hotkey('ctrlleft', 'v')


def example1():
    # 储存图片坐标
    dict1 = {}

    # 找到微信图标并双击
    dict1['wechat_icon'] = img_check('../img/wx_icon.png')
    if dict1['wechat_icon'][0] == 0 and dict1['wechat_icon'][1] == 0:
        print('没找到wechat_icon')
        sys.exit()
    else:
        pyautogui.moveTo(dict1['wechat_icon'][0], dict1['wechat_icon'][1], 1)
        pyautogui.doubleClick()

    time.sleep(2)

    dict1['search_icon'] = img_check('../img/search_icon.png')
    if dict1['search_icon'][0] == 0 and dict1['search_icon'][1] == 0:
        print('没找到搜索框')
        sys.exit()
    else:
        search_friends(dict1['search_icon'][0], dict1['search_icon'][1], "六一二")

    time.sleep(2)

    # 加载发送文本
    list_msg = []
    with open('../wx_auto_msg/msg.txt', encoding='utf8', mode='r') as msg_file:
        for msg in msg_file:
            list_msg.append(str(msg).strip('\n'))
    msg_file.close()
    print(list_msg)

    count = 0
    while True:
        if count > len(list_msg) - 1:
            x, y = img_check('../img/search_icon.png')
            if x > 0 and y > 0:
                pyautogui.hotkey('altleft', 'f4')
            break
        else:
            type_msg(list_msg[count])
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.hotkey('altleft', 'f4')
            time.sleep(5)
            count = count + 1
            pyautogui.moveTo(dict1['wechat_icon'][0], dict1['wechat_icon'][1], 0.5)
            pyautogui.doubleClick()
            time.sleep(1)
            search_friends(dict1['search_icon'][0], dict1['search_icon'][1], "六一二")


def example2():
    img_match.img_match_by_RGB2gray('../img/sc.png', '../img/wx_icon.png')


def main():
    example1()


if __name__ == '__main__':
    main()
