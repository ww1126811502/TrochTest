# -*- coding = utf-8 -*-
# @Time : 2022/8/19 21:17
# @Author : 牧川
# @File : comb_name.py
import pandas as pd
from pandas import DataFrame
import codecs


def handleEncoding(original_file, newfile):
    # newfile=original_file[0:original_file.rfind(.)]+'_copy.csv'
    f = open(original_file, 'rb+')
    content = f.read()  # 读取文件内容，content为bytes类型，而非string类型
    source_encoding = 'utf-8'
    #####确定encoding类型
    try:
        content.decode('utf-8').encode('utf-8')
        source_encoding = 'utf-8'
    except:
        try:
            content.decode('gbk').encode('utf-8')
            source_encoding = 'gbk'
        except:
            try:
                content.decode('gb2312').encode('utf-8')
                source_encoding = 'gb2312'
            except:
                try:
                    content.decode('gb18030').encode('utf-8')
                    source_encoding = 'gb18030'
                except:
                    try:
                        content.decode('big5').encode('utf-8')
                        source_encoding = 'gb18030'
                    except:
                        try:
                            content.decode('cp936').encode('utf-8')
                            source_encoding = 'cp936'
                        except:
                            try:
                                content.decode('UTF-16 LE').encode('utf-8')
                                source_encoding = 'UTF-16 LE'
                            except:
                                content.decode('UTF-16 BE').encode('utf-8')
                                source_encoding = 'UTF-16 BE'
    print(source_encoding)
    f.close()

    #####按照确定的encoding读取文件内容，并另存为utf-8编码：
    block_size = 4096
    with codecs.open(original_file, 'r', source_encoding) as f:
        with codecs.open(newfile, 'w', 'utf-8') as f2:
            while True:
                content = f.read(block_size)
                if not content:
                    break
                f2.write(content)


def get_result(filename):
    # try:
    #     df = pd.read_csv(filename, encoding='utf-8')
    # except BaseException:
    #     df = pd.read_csv(filename, encoding='cp950')
    # print(df.head())
    df = pd.read_csv(filename, encoding='UTF-8', sep='\t')
    print(df.head())
    type1 = df.groupby(['类型【1长度9、2长度6、3长度3】']).get_group("1")['姓名称谓']
    type2 = df.groupby(['类型【1长度9、2长度6、3长度3】']).get_group("2")['姓名称谓']
    temp_list = []
    fist = type1.values.tolist()
    second = type2.values.tolist()
    for m in fist:
        for n in second:
            temp_list.append(m + n)

    df_res = DataFrame(temp_list)
    df_res.to_csv(f'{filename}_result.csv', sep=' ', index=0, encoding="utf_8_sig", header=0)
    print('保存成功！')


if __name__ == '__main__':
    handleEncoding('RoleNameCN.csv', 'RoleNameCN-2.csv')
    get_result('RoleNameCN-2.csv')
