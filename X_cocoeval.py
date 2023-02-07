"""
对pycocotools中cocoeval.py中部分函数的注释
注: 此代码不会被运行, 仅用于注释
"""

def evaluateImg(self, imgId, catId, aRng, maxDet):
    '''
    对单幅图像(imgId)某一类(catId)的检测结果进行特定评估(给定aRng, maxDet)
    :return: dict (single image results)
    '''
    p = self.params
    if p.useCats:
        # self._gts:
        # {
        #   (image_id_1, class_1   ): [{...}, {...}, ...],
        #   ...
        #   (image_id_1, class_M  ): [{...}, {...}, ...],
        #   ...
        #   (image_id_N, class_1   ): [{...}, {...}, ...],
        #   ...
        #   (image_id_N, class_M  ): [{...}, {...}, ...]
        # } 
        # N=batch_size, M个类别. 例如4幅图像20个类别, 则len(self._gts) = 4 * 20 = 80
        # gt = self._gts[0, 20]:
        # [
        # {'image_id': 0, 'bbox': [34.0, 11.0, 414.0, 282.0], 'category_id': 20, 'area': 116748.0, 'is_crowd': 0, 'id': #  1, 'ignore': 0},
        #  ...
        # {'image_id': 0, 'bbox': [14.0, 21.0, 321.0, 117.0], 'category_id': 20, 'area': 321180.0, 'is_crowd': 0, 'id': #  2, 'ignore': 0},
        # ]
        gt = self._gts[imgId,catId]  # ground truth: image_id=0的图像所包含的class=20的标注
        dt = self._dts[imgId,catId]  # detection   : image_id=0的图像所包含的class=20的标注
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    if len(gt) == 0 and len(dt) ==0:
        return None

    for g in gt:
        # 如果标注g不在aRng指定的范围内, 如aRng=[1024, 9216]时, g['area']<1024或>9216则设置为g['_ignore'] = 1
        if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    # np.argsort([5, 2, 1, 4, 9]) -> [2, 1, 3, 0, 4]
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]  # 将'_ignore'=0的gt排在前面
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]  # 将'score'大的dt排在前面, 取前maxDet个
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    # self.ious: {(0, 1): [], (0, 2): [], ...}
    # 猜测: ious[imgId, catId]是imgId这幅图像中, catId这个类别的所有g和d之间的两两IoU矩阵, 如下
    # [[g1-d1, g2-d1, ..., gN-d1],
    #  [g1-d2, g2-d2, ..., gN-d2],
    #   ...
    #  [g1-dM, g2-dM, ..., gN-dM]]
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

    T = len(p.iouThrs)  # 10: p.iouThrs = [0.5, 0.55, ..., 0.95]
    G = len(gt)  # ground_truth数目
    D = len(dt)  # detection数目
    gtm  = np.zeros((T,G))  # 例: gtm[0, :]: 当p.iouThrs[0]时, 所有ground truth所匹配的detection的id, 未匹配则为0
    dtm  = np.zeros((T,D))  # 例: dtm[0, :]: 当p.iouThrs[0]时, 所有detection所匹配的ground truth的id, 未匹配则为0
    gtIg = np.array([g['_ignore'] for g in gt])  # [0, 0, ..., 0, 1, ..., 1]
    dtIg = np.zeros((T,D))  # 例: dtIg[0, :]: 当p.iouThrs[0]时, 所有detection能够匹配则为true, 否则为false
    if not len(ious)==0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t,1-1e-10])
                m   = -1
                # 遍历gt, 寻找与d的IoU最大的g.
                # 如果寻找到满足条件的g, 则将g的id放入dtm, 同时将d的id放入gtm.
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    # 条件1: gtm[tind, gind]>0表示当前元素已经被匹配过.
                    # 条件2: not iscrowd[gind]表示当前被匹配的元素并不和其它目标重叠
                    if gtm[tind,gind]>0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    # gtIg[m]表示上一个元素, gtIg[gind]表示当前元素.
                    # 条件1: m>-1表示只能从gtIg的第2个元素开始比较
                    # 条件2: gtIg[m] == 0 and gtIg[gind] == 1, 表示当前的元素已经是被忽略的元素
                    # 因为gtIg = [0, ..., 0, 1, ..., 1], 所以'_ignore'=0的元素已经遍历完
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made

                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=ious[dind,gind]
                    m=gind
                # if match made store id of match for both dt and gt
                if m ==-1:
                    continue
                dtIg[tind,dind] = gtIg[m]  # 是否匹配到
                dtm[tind,dind]  = gt[m]['id']
                gtm[tind,m]     = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
    # store results for given image and category
    return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }

def accumulate(self, p = None):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    在不同的 iouThr, recThr, catId, areaRng, maxDet 下, 
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = self.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    # 10: [0.5, 0.55, ..., 0.95] 10个级别的IoU
    T           = len(p.iouThrs)
    # 101: [0, 0.01, ..., 1.0] 101个概率阈值
    R           = len(p.recThrs)
    # 20: [1, 2, ..., 20]
    K           = len(p.catIds) if p.useCats else 1
    # 4: [0, 10000000000]    -> 无限制
    #    [0, 1024]           -> 小目标
    #    [1024, 9216]        -> 中目标
    #    [9216, 10000000000] -> 大目标
    A           = len(p.areaRng)
    # 3: [1, 10, 100] 每张图片最大检测目标个数
    M           = len(p.maxDets)
    precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    recall      = -np.ones((T,K,A,M))
    scores      = -np.ones((T,R,K,A,M))

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)   # 已经被评估过的图像的id
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        # 假设(1)20个class; (2)4个areaRng; (3)已经推断了12幅图像, A0=4, I0=4: 
        # 则Nk = [0, 48, 96, 144, 192, ..., 912]
        # 每个类别下有4 * 12个结果, 总计960个结果.
        Nk = k0*A0*I0
        for a, a0 in enumerate(a_list):
            # Na = [0, 12, 24, 36]
            # 每个areaRng下有12个结果
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                # i = [0, 1, 2, 3, 4, ..., 11] -> 12幅图像
                # (Nk + Na + i)完成对self.evalImgs的遍历
                # E: 全部已经评估过的12幅图像的结果, 这些结果category_id相同, areaRng相同
                E = [self.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]  # Q: 可能存在图像不存在某个类别或是areaRng?
                if len(E) == 0:
                    continue
                # 1.e['dtScores']是一个1维list: [0.1162128895521164, 0.10159151256084442]
                # 2.e['dtScores']中的个数可能不足maxDet, 不足时会返回全部元素
                # 3.dtScores是1维的array
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]  # 从大到小的顺序排序

                # 提示: 
                # e['dtMatches']行为iou_thrs, 10行代表[0.5, 0.55, 0.6, ..., 0.95], 列代表detection匹配到的ground truth的id
                # dtm: 每一行都把匹配到的ground truth按照inds重新排序
                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                # dtIg: 每一行都把'是否被忽略'按照inds重新排序
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg==0 )  # ground truth的数目
                if npig == 0:
                    continue
                # tps代表真阳性?fps代表假阳性?
                # tps: detection拥有匹配到的ground truth, 而且这个detection本身也是符合限定条件的
                tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                # fps: detection没有匹配到的ground truth, 但是这个detection本身却符合限定条件
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                # x -> [1, 3, 5, 7], np.cumsum(x) -> [1, 4, 9, 16]沿指定维度累加元素
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    # Recall = TP / (TP + FN), 召回的真阳性占全部真阳性(TP + FN)的比例
                    rc = tp / npig
                    # Precision = TP / (TP + FP), 真阳性占判定为阳性(TP + FP)的比例
                    pr = tp / (fp+tp+np.spacing(1))  # np.spacing(1)防止除零
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t,k,a,m] = rc[-1]  # Q: 为何只用最大的召回率?
                    else:
                        recall[t,k,a,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        # 2023年2月6日 15点4分
                        # 猜测是为了平滑precision, 假设:
                        # tp: [1, 2, 3, 4]
                        # fp: [1, 1, 2, 3]
                        # pr: [1/2, 2/3, 3/5, 4/7]
                        # 2/3 > 1/2, 所以pr变成[2/3, 2/3, 3/5, 4/7]
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]
                    
                    # 这一句似乎没有起到作用
                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t,:,k,a,m] = np.array(q)
                    scores[t,:,k,a,m] = np.array(ss)
    self.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall':   recall,
        'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format( toc-tic))