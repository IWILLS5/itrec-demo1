
class EvaluationMetrics():
    def __init__(self, num_label):
        self.num_label = num_label

        self.TP = [0] * self.num_label 
        self.T_sort = [0] * self.num_label #真实是该样本
        self.P_sort = [0] * self.num_label #预测为该样本

        self.precision = [0] * self.num_label
        self.recall = [0] * self.num_label
        self.F1 = [0] * self.num_label

    def reset(self):
        self.TP = [0] * self.num_label 
        self.T_sort = [0] * self.num_label #真实是该样本
        self.P_sort = [0] * self.num_label #预测为该样本
        self.precision = [0] * self.num_label
        self.recall = [0] * self.num_label
        self.F1 = [0] * self.num_label

    #添加一个batch的数据
    def add_batch(self, trues, predictions):
        for t, p in zip(trues, predictions):
            if t == p:
                self.TP[t] += 1
            self.T_sort[t] += 1
            self.P_sort[p] += 1
    #计算每个类别的precision、recall、F1
    def em_count(self):
        
        for i in range(self.num_label):
            tp = self.TP[i]
            fp = self.P_sort[i] - tp
            fn = self.T_sort[i] - tp

            self.precision[i] = tp / (tp + fp) if (tp + fp) else 0.0
            self.recall[i]    = tp / (tp + fn) if (tp + fn) else 0.0
            p, r = self.precision[i], self.recall[i]
            self.F1[i] = 2 * p * r / (p + r) if (p + r) else 0.0
    
    def get_Macro_data(self):
        self.em_count()
        precision = sum(self.precision) / self.num_label
        recall = sum(self.recall) / self.num_label
        f1 = sum(self.F1) / self.num_label
        return precision, recall, f1

if __name__ == "__main__":
    em = EvaluationMetrics(15)
    em.add_batch([1 ,2, 3], [2, 3, 4])