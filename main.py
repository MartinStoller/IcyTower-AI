import mss
import neat
import numpy as np
import cv2
import time
import collections
from pynput.keyboard import Key, Controller
import csv
import keyboard
import pyautogui
import psutil
import os
import pytesseract
import pickle

mss_instance = mss.mss()  # needed to take rapid screenshots
keyboard1 = Controller()  # needed to simulate pressing buttons
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LOWER_RANGE_AGENT = np.array([8, 137, 197], np.uint8)  # color detection of agent
UPPER_RANGE_AGENT = np.array([12, 140, 199], np.uint8)

LOWER_RANGE_STONE = np.array([115, 17, 237], np.uint8)  # color detection stone-platform
UPPER_RANGE_STONE = np.array([130, 42, 255], np.uint8)

lower_range_gameover = np.array([102, 161, 213], np.uint8)
upper_range_gameover = np.array([104, 168, 215], np.uint8)

x_agent_queue = collections.deque([120, 120])
y_agent_queue = collections.deque([406, 406])

death_type = 2  # if 0, then death by timeout, if 1 then death by falling


def close_icytower():
    for process in (process for process in psutil.process_iter() if process.name() == "icytower15.exe"):
        process.kill()


def start_new_game(deathtype):
    time.sleep(0.5)
    if deathtype == 1:
        time.sleep(2)
        keyboard1.press(Key.space)
        time.sleep(3)
        keyboard1.release(Key.space)
        time.sleep(0.5)
        keyboard1.press(Key.space)
        time.sleep(2)
        keyboard1.release(Key.space)
        lower_range_menu = np.array([23, 25, 247], np.uint8)
        upper_range_menu = np.array([25, 27, 249], np.uint8)
        screenshot = mss_instance.grab({
            "left": 719,
            "top": 281,
            "width": 484,
            "height": 474
        })
        img = np.array(screenshot)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_menu = cv2.inRange(hsv_img, lower_range_menu, upper_range_menu)
        mask_menu2 = cv2.findNonZero(mask_menu)

        while mask_menu2 is not None and len(mask_menu2) > 3_000:
            time.sleep(1)
            keyboard1.press(Key.space)
            time.sleep(0.3)
            keyboard1.release(Key.space)
            time.sleep(0.5)
            keyboard1.press(Key.space)
            time.sleep(0.3)
            keyboard1.release(Key.space)

            screenshot = mss_instance.grab({
                "left": 719,
                "top": 281,
                "width": 484,
                "height": 470
            })
            img = np.array(screenshot)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_menu = cv2.inRange(hsv_img, lower_range_menu, upper_range_menu)
            mask_menu2 = cv2.findNonZero(mask_menu)
    elif deathtype == 0:
        keyboard1.press(Key.esc)
        time.sleep(0.1)
        keyboard1.release(Key.esc)
        time.sleep(1)
        keyboard1.press(Key.esc)
        time.sleep(0.1)
        keyboard1.release(Key.esc)
        time.sleep(1.2)
        keyboard1.press(Key.space)
        time.sleep(0.15)
        keyboard1.release(Key.space)
        time.sleep(0.8)
        keyboard1.press(Key.space)
        time.sleep(0.15)
        keyboard1.release(Key.space)
        keyboard1.press(Key.space)
        time.sleep(0.15)
        keyboard1.release(Key.space)
        time.sleep(0.8)
        keyboard1.press(Key.space)
        time.sleep(0.24)
        keyboard1.release(Key.space)
    else:
        print("Start_new_game() did not work properly!")


def restart_icytower():
    close_icytower()
    time.sleep(5)
    os.startfile("C:\games\icytower1.5\icytower15.exe")
    time.sleep(8)
    keyboard1.press(Key.space)
    time.sleep(0.2)
    keyboard1.release(Key.space)
    time.sleep(0.8)
    keyboard1.press(Key.space)
    time.sleep(0.2)
    keyboard1.release(Key.space)
    time.sleep(0.8)
    keyboard1.press(Key.space)
    time.sleep(0.2)
    keyboard1.release(Key.space)
    time.sleep(0.8)
    keyboard1.press(Key.space)
    time.sleep(0.25)
    keyboard1.release(Key.space)
    pyautogui.moveRel(790, 230)
    time.sleep(1)


def take_action(action_jump, action_lr):
    if action_jump == 0:
        keyboard1.release("j")
    else:
        keyboard1.press("j")

    if action_lr == 0:
        keyboard1.release("d")
        keyboard1.press("a")
    else:
        keyboard1.release("a")
        keyboard1.press("d")


def update_platformdict(dict, key, x1, x2):  # update platform_dict, so we get max length of platform
    if x1 < dict[key][0]:
        dict[key][0] = x1
    if x2 > dict[key][1]:
        dict[key][1] = x2


def get_platforms(hsv_img):
    mask_platform_stone = cv2.inRange(hsv_img, LOWER_RANGE_STONE, UPPER_RANGE_STONE)
    lines = cv2.HoughLinesP(image=mask_platform_stone, rho=1, theta=np.pi / 180, threshold=100,
                            minLineLength=90, maxLineGap=50)
    platform_dict = {}
    # Merge close lines:
    if lines is not None:
        N = lines.shape[0]
        for i in range(N):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            if not any(item in range(y1 - 8, y1 + 8) for item in platform_dict):
                platform_dict[y1] = [x1, x2]
            else:
                for key in platform_dict:
                    if key in range(y1 - 8, y1 + 8):
                        update_platformdict(dict=platform_dict, key=key, x1=x1, x2=x2)
    return platform_dict


def get_agent_speed(x, y):
    x_agent_queue.appendleft(x)
    y_agent_queue.appendleft(y)
    x_agent_queue.pop()
    y_agent_queue.pop()
    vx = x_agent_queue[1] - x_agent_queue[0]
    vy = y_agent_queue[1] - y_agent_queue[0]
    return vx, vy


def take_screenshots():
    screenshot = mss_instance.grab({
                "left": 719,
                "top": 281,
                "width": 484,
                "height": 474
            })
    img = np.array(screenshot)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv_img


def get_agent_coordinates(hsv_img):
    mask_agent = cv2.inRange(hsv_img, LOWER_RANGE_AGENT, UPPER_RANGE_AGENT)
    coord_agent = cv2.findNonZero(mask_agent)
    if coord_agent is None or len(coord_agent) < 7:
        agent_x = 0
        agent_y = 0
    else:
        agent_x = int(sum(x[0][0] for x in coord_agent) / len(
            coord_agent))
        agent_y = int(sum(x[0][1] for x in coord_agent) / len(coord_agent))
    return agent_x, agent_y


def get_platforms_2(platform_dict):
    platform_indices = []
    for key in platform_dict:
        platform_indices.append(key)
    l = len(platform_indices)
    for i in range(6 - l):
        platform_indices.append(i)
        platform_dict[i] = [0, 0]
    platform_indices = sorted(platform_indices)
    px00 = platform_dict[platform_indices[0]][0]
    px10 = platform_dict[platform_indices[1]][0]
    px20 = platform_dict[platform_indices[2]][0]
    px30 = platform_dict[platform_indices[3]][0]
    px40 = platform_dict[platform_indices[4]][0]
    px50 = platform_dict[platform_indices[5]][0]
    py0 = platform_indices[0]
    py1 = platform_indices[1]
    py2 = platform_indices[2]
    py3 = platform_indices[3]
    py4 = platform_indices[4]
    py5 = platform_indices[5]
    px01 = platform_dict[platform_indices[0]][1]
    px11 = platform_dict[platform_indices[1]][1]
    px21 = platform_dict[platform_indices[2]][1]
    px31 = platform_dict[platform_indices[3]][1]
    px41 = platform_dict[platform_indices[4]][1]
    px51 = platform_dict[platform_indices[5]][1]
    return px00, px10, px20, px30, px40, px50, px01, px11, px21, px31, px41, px51, py0, py1, py2, py3, py4, py5


def detect_terminal_state(hsv_img, time_before50):
    mask_gameover = cv2.inRange(hsv_img, lower_range_gameover, upper_range_gameover)
    gameover_pixels = cv2.findNonZero(mask_gameover)
    ts = False
    dt = 2
    if gameover_pixels is not None:
        if len(gameover_pixels) > 3_000:
            ts = True
            dt = 1
    elif time_before50 > 120:
        ts = True
        dt = 0
    return ts, dt


def detect_action():
    if keyboard.is_pressed('a'):
        keyboard.release("d")
        action_lr = 0
    else:
        keyboard.press("d")
        action_lr = 1
    if keyboard.is_pressed("j"):
        action_jump = 1
    else:
        action_jump = 0
    return action_lr, action_jump


def get_score(img, score_queue, score_reward_queue, reward, passed_platforms):
    gray = cv2.cvtColor(img[444:, 24:150], cv2.COLOR_BGR2GRAY)
    mask_score = cv2.inRange(gray, 251, 255)
    thresh_score = cv2.threshold(mask_score, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gaus_img_score = cv2.GaussianBlur(thresh_score, (3, 3), 0)
    score = pytesseract.image_to_string(gaus_img_score, lang='font_name2',
                                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
    score_queue.appendleft(score)
    score_queue.pop()
    try:
        for i in range(8):
            if score_queue[i] == score_queue[i + 1]:
                score = score_queue[i]
                score_reward_queue.appendleft(score)
                score_reward_queue.pop()
                break
        reward += (int(score_reward_queue[0]) - int(score_reward_queue[1]))
        if reward < -20:
            reward = 0
        elif reward > 2000 and passed_platforms < 40:
            reward = 0
        elif reward > 1390 and passed_platforms < 30:
            reward = 0
        elif reward > 800 and passed_platforms < 24:
            reward = 0
        elif reward > 390 and passed_platforms < 17:
            reward = 0
        elif reward > 190 and passed_platforms < 8:
            reward = 0
    except:
        print("Warning: Problems with Score detection")
        restart_icytower()
    return reward, score


def eval_genomes(genomes, config):
    games_played = 0
    score = 0
    passed_platforms = 0
    highest_platform_queue = collections.deque([33, 33, 33, 33])
    score_queue = collections.deque([0, 0, 0, 0, 0, 0, 0, 0, 0])  # use to filter out wrongly detected numbers
    score_reward_queue = collections.deque([0, 0])  # use to compare current score with previous score for reward
    action_lr = 0  # 0 = agent moves left/ holds left keyboard key; 1 = right -> hold right keyboard key
    action_jump = 0  # 0 = agent does not jump; 1 = press j (= jump) -> 2nd output cell

    time.sleep(2)

    # evaluate fitness of each genome
    for genome_id, genome in genomes:
        time_before50 = 0
        genome.fitness = 0  # fitness is the sum of rewards our avatar received at death
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # set up NN for our genome

        terminal_state = False

        while terminal_state is False:
            start_time = time.time()
            reward = 0
            take_action(action_jump=action_jump, action_lr=action_lr)
            # detect new state and the reward our action(from the previous iteration) caused:
            img, hsv_img = take_screenshots()
            if int(score) <= 40:
                time_before50 += 1
            else:
                time_before50 = 0
            x,y = get_agent_coordinates(hsv_img)
            vx, vy = get_agent_speed(x, y)
            platform_dict = get_platforms(hsv_img)
            px00, px10, px20, px30, px40, px50, px01, px11, px21, px31, px41, px51, py0, py1, py2, py3, py4, py5 = get_platforms_2(platform_dict)
            terminal_state, death_type = detect_terminal_state(hsv_img, time_before50)
            elapsed_time = time.time() - start_time
            vx = int(vx / elapsed_time)
            vy = int(vy / elapsed_time)
            input_data = [x, y, vx, vy, x - px00, x - px10, x - px20, x - px30, x - px40, x - px50, y - py0, y - py1,
                          y - py2, y - py3, y - py4, y - py5, px01 - x, px11 - x, px21 - x, px31 - x, px41 - x, px51 - x]
            highest_platform_queue.appendleft(py0)
            highest_platform_queue.pop()
            if (highest_platform_queue[0] + 20) < highest_platform_queue[1]:
                passed_platforms += 1
            reward, score = get_score(img, score_queue, score_reward_queue, reward, passed_platforms)
            # ask net what action to choose:
            output = net.activate(tuple(input_data))  # output will be the output of our NN what action we take
            genome.fitness += reward  # reward detected after action was taken
            if genome.fitness > 8_500:
                genome.fitness = 0
            if output[0] >= 0:
                action_lr = 1
            else:
                action_lr = 0
            if output[1] >= 0:
                action_jump = 1
            else:
                action_jump = 0

            if terminal_state is True:
                # make sure that no more keys are pressed
                keyboard1.release("a")
                keyboard1.release("j")
                keyboard1.release("d")
                keyboard1.release(Key.space)
                games_played += 1
                time.sleep(0.2)
                start_new_game(death_type)
                # reset parameters:
                passed_platforms = 0
                score = 0
                score_queue = collections.deque([0, 0, 0, 0, 0, 0, 0, 0, 0])
                score_reward_queue = collections.deque([0, 0])
                action_lr = 0
                action_jump = 0
                print("games played: " + str(games_played) + "/" + str(len(genomes)))
                print("reward: " + str(genome.fitness))
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)

                with open("scorelogger_lastone.csv", "a+", newline="") as log_csv:
                    writer = csv.writer(log_csv, dialect="excel")
                    row_contents = [int(games_played), genome_id, str(current_time), genome.fitness]
                    writer.writerow(row_contents)
                time.sleep(1)
                if games_played % 50 == 0:
                    restart_icytower()
                time.sleep(1)
                break


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-")

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.Checkpointer(generation_interval=1, time_interval_seconds=1800))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 40)
    filename = 'neat_winner2.sav'
    pickle.dump(winner, open(filename, 'wb'))
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    time.sleep(2)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)