import pandas as pd
import datetime


# Connect L_{k-1} to C_k
def connect_string(x, ms):
    # map(a, b) -> 对b中的所有元件施行a，即如果b为[1, 2, 3], a为函数，则map(a, b) -> [a(1), a(2), a(3)]
    # lambda 为python语言中一种快速定义函数的方法；lambda x: x + 1 意思是接收输入变量x，返回x+1
    # 此处若names为["A", "B,C", "D"], 则输出为[["A"], ["B", "C"], ["D"]]
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    # 如果：前部分相同而最后一个不同，比如[A, B, C]和[A, B, D]就满足
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
    return r
 # 最终的r即获得所有的排列组合
    # 例如 [['Beer', 'Cola'], ['Beer', 'Diaper'], ['Beer', 'Egg'], ['Beer', 'Ham']]


def trans(data):#数据清洗，转换
    column_names = data.columns.tolist()# 获取所有的列表抬头（列表名称）-> ['tid', 'items']
    tid = list(data[column_names[0]])# 获取tid列的所有数据 -> [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    item = list(data[column_names[1]])# 获取items列的所有数据 -> ['Cola', 'Egg', 'Ham', 'Ham', 'Diaper', 'Beer']
    tids, items = [], []
    # t交易编号，i物品名称
    for t, i in zip(tid, item):
        i = i.split(',')# 以防物品名称部分包含多项物品，以","作为分隔
        items += i# 将本项的所有的物品计入物品列表
        tids += [t] * len(i)# 将本项的对应的交易计入交易编号记录列表
    data = pd.DataFrame({'tid': tids, 'item': items})
    return data
#1 cola,egg
#1 cola
#1 egg

# Fine Association Rule
def find_rule(d, support, confidence, ms = u'--'):
    """
        :param data: 数据格式：矩阵形式的，只有1/0数据的，pandas.core.frame.DataFrame
        :param support: 最小支持度
        :param confidence: 最小置信度
        :param ms: 字符间连接符号
        :return: 数据格式：pandas.core.frame.DataFrame
        """

    result = pd.DataFrame(index=['support', 'confidence'])  # 初始化最终结果的数据结构

    # 获得【支持度序列】：每个物品出现的频次（对于交易数量来说）
    # data.sum() -> 所有物品的交易数量（例如：Cola：3.0; Egg: 1.0）
    # len(data) -> 交易的数量
    # 备注：起名规范 -> support（支持度），series（序列）
    support_series = 1.0 * d.sum() / len(d)  # 支持度序列

    # 根据支持度进行筛选（筛选出满足最低支持度要求的，即最低出现频次要求的物品）
    # 例：['Beer', 'Cola', 'Diaper', 'Egg', 'Ham'],
    column = list(support_series[support_series > support].index)
    k = 0

    while len(column) > 1:# 当至少确保有有效的物品列表（即至少有物品的出现频次满足要求）时
        k = k+1# k 代表搜索的次数
        print(u'\n正在进行第%s次搜索...' %k)
        column = connect_string(column, ms)
        print(u'数目：%s...' %len(column))

        # 物品组合的支持度的计算函数，其中这里的names用于选择data中对应物品的列表
        # 这里的prod意思为乘法，即选择出来的物品交易相乘，获得同时包括所有所选物品的交易条目
        sf = lambda i: d[i].prod(axis=1, numeric_only = True) #新一批支持度的计算函数

        # 计算物品组合的支持度（组合出现频次）
        # 这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化
        d_2 = pd.DataFrame(list(map(sf,column)), index = [ms.join(i) for i in column]).T

        support_series_2 = 1.0*d_2[[ms.join(i) for i in column]].sum()/len(d) # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index) # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)# 前后的支持度列表结合在一起
        column2 = []# column2将包含全部组合可能性的物品名称

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j]+i[j+1:]+i[j:j+1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 初始化置信度序列（confidence置信度）（series序列）

        for i in column2:  # 计算置信度序列
            # 每个组合的置信度 = 组合支持度 / 单个物品的支持度
            # 例如：组合为[A, B], 即置信度 = 支持度([A,B]) / 支持度([A])
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))]/support_series[ms.join(i[:len(i)-1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 筛选符合置信度要求的组物品组合
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    index_list = list(result.index)
    X=[x.replace(',' + x.split(',')[-1], '') for x in index_list]  # 提取规则前项
    Y=[x.split(',')[-1] for x in index_list]  # 提取规则后项
    result['X'] = X
    result['Y'] = Y
    result.index = range(1, len(result) + 1)  # 重置index
    result = result[['X', 'Y', 'support', 'confidence']]
    return result


def operate(data, support=0.02, confidence=0.2):
    """
    :param data: 数据
    :param support: 最小支持度
    :param confidence: 最小置信度
    :return:
    """
    # ----------开始计算----------------
    print('开始计算!')
    t1 = datetime.datetime.now()
    data = trans(data)  # 数据转换
    column_names = data.columns.tolist()  # 获取数据字段名称
    print('转换原始数据至0-1矩阵...')
    # 去重 (pandas数据形式自带的去重机制)
    # --> subset: 查重覆盖的列表范围
    # --> keep: 遇到重复信息的保留机制
    # --> inplace: 是否改动自身变量
    data.drop_duplicates(subset=column_names, keep='first', inplace=True)  # 去重
    data['是否购买'] = 1  # 增加一列值都为1
    data = data.pivot(index=column_names[0], columns=column_names[1], values='是否购买')  # 数据透视，1代表有购买
    sale_num = len(data)# 交易数量（即unique(tid)的个数）
    data.index = range(sale_num)   # 重置索引（例：原索引index为1～4，变更后为0～3）
    data = data.fillna(0)  # 实现矩阵转换，空值用0填充
    print('转换完毕。')
    result = find_rule(data, support, confidence, ',')  # 用apriori进行规则挖掘
    print('销售记录数:', sale_num)
    t2 = datetime.datetime.now()
    print('Spend Ts:', t2 - t1)
    print('计算结束!')
    return result