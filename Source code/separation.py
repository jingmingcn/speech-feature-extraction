from pyannote.audio import Pipeline
from pydub import AudioSegment
import os


def isexist(name, path=None):
    '''
    :param name: 需要检测的文件或文件夹名
    :param path: 需要检测的文件或文件夹所在的路径，当path=None时默认使用当前路径检测
    :return: True/False 当检测的文件或文件夹所在的路径下有目标文件或文件夹时返回Ture,
            当检测的文件或文件夹所在的路径下没有有目标文件或文件夹时返回False
    '''
    if path is None:
        path = os.getcwd()
    if os.path.exists(path + '/' + name):
        print("Under the path: " + path + '\n' + name + " is exist")
        return True
    else:
        if (os.path.exists(path)):
            print("Under the path: " + path + '\n' + name + " is not exist")
        else:
            print("This path could not be found: " + path + '\n')
        return False


offline_vad = Pipeline.from_pretrained("./offline/config.yaml")
diarization = offline_vad("../data/test1.wav")  # 此时的文件路径应该改为变量path1.file1
# diarization = pipeline("audio.wav", num_speakers=2) 准确的发言者数量
# diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5) 发言者数量上下限


# dump the diarization output to disk using RTTM format
# with open("./rttm/outline.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
#  print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    newAudio = AudioSegment.from_wav("../data/test1.wav")
    start = turn.start
    end = turn.end
    newAudio = newAudio[int(start * 1000):int(end * 1000)]
    if not isexist(name=speaker + '.wav', path='.'):
        newAudio.export(speaker + '.wav', format="wav")
    else:
        newAudio.export('temp.wav', format="wav")
        combined_sounds = AudioSegment.from_wav(speaker + ".wav")
        temp = AudioSegment.from_wav("temp.wav")
        combined_sounds = combined_sounds + temp
        # os.remove(speaker + '.wav')
        combined_sounds.export(speaker + '.wav', format="wav")
        os.remove('temp.wav')


# # cut and combine
# newAudio = AudioSegment.from_wav("./testaudio/finaltest.wav")  # 文件改为file1
# # Start timestamps for Speaker 0
# start1 = [0.498]
# # End timestamps for Speaker 0
# end1 = [5.881]
#
# for i in range(len(start1)):
#     newAudio = AudioSegment.from_wav("exp.wav")
#     newAudio = newAudio[int(start1[i] * 1000):int(end1[i] * 1000)]
#     newAudio.export('output1.wav', format="wav")
# # combine
# combined_sounds = AudioSegment.from_wav("output1.wav")
# combined_sounds = combined_sounds
# combined_sounds.export('finaltest.wav', format="wav")
