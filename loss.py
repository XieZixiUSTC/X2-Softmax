import torch


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, parameters):
        super(ArcFace, self).__init__()
        self.s = parameters[0]
        self.margin = parameters[1]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, parameters):
        super(CosFace, self).__init__()
        self.s = parameters[0]
        self.m = parameters[1]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class ElasticARC(torch.nn.Module):
    def __init__(self, parameters):
        super(ElasticARC, self).__init__()
        self.s = parameters[0]
        self.mean = parameters[1]
        self.sigma = parameters[2]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()

            elastic = torch.normal(self.mean, self.sigma, [len(index)]).to(device='cuda:0')

            final_target_logit = target_logit + elastic
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class ElasticCOS(torch.nn.Module):
    def __init__(self, parameters):
        super(ElasticCOS, self).__init__()
        self.s = parameters[0]
        self.mean = parameters[1]
        self.sigma = parameters[2]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        elastic = torch.normal(self.mean, self.sigma, [len(index), 1]).to(device='cuda:0')
        final_target_logit = target_logit - elastic
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class SOFTMAX(torch.nn.Module):
    def __init__(self, parameters):
        super(SOFTMAX, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        return logits


class NORMFACE(torch.nn.Module):
    def __init__(self, parameters):
        super(NORMFACE, self).__init__()
        self.s = parameters[0]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        return logits * self.s


class X2Softmax(torch.nn.Module):
    def __init__(self, parameters):
        super(X2Softmax, self).__init__()
        self.s = parameters[0]
        """
        # a * x **2 + b * x + c = 0
        self.a = parameters[1]
        self.b = parameters[2]
        self.c = parameters[3]
        """
        # f = a * (x - h) ** 2 + k
        self.a = parameters[1]
        self.h = parameters[2]
        self.k = parameters[3]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            # final_target_logit = self.a * target_logit ** 2 + self.b * target_logit + self.c
            final_target_logit = self.a * (target_logit - self.h) ** 2 + self.k
            logits[index, labels[index].view(-1)] = final_target_logit

        logits = logits * self.s
        return logits


class DYNARCLOSS(torch.nn.Module):
    def __init__(self, parameters):
        super(DYNARCLOSS, self).__init__()
        self.s = parameters[0]
        self.k1 = parameters[1]
        self.k2 = parameters[2]
        self.k3 = parameters[3]

    def margin(self, labels, weight):
        # labels m, 1
        # weight c, d
        w_labels = weight[labels]  # m, d
        w_costheta = (w_labels @ weight.t()).clamp(-1, 1)  # m, c
        index = torch.where(labels != -1)[0]
        w_costheta[index, labels] -= 2
        costheta_wn_wyi = torch.max(w_costheta, 1)[0]
        theta = torch.arccos(costheta_wn_wyi)  # angle between wn and wyi
        margin = torch.relu(theta - self.k1) * self.k2 + self.k3 + self.smooth(theta)
        return margin

    def smooth(self, theta):
        smooth = 0.03 * self.k3 / (1 + torch.pow(torch.abs(theta - self.k1) * 20, 1.1))
        return smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, weight_norm):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()

            margin = self.margin(labels, weight_norm)
            final_target_logit = target_logit + margin

            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()

        logits = logits * self.s
        return logits


class MAGFACELOSS(torch.nn.Module):
    def __init__(self, parameters):
        super(MAGFACELOSS, self).__init__()
        self.s = parameters[0]
        self.l_m = parameters[1]
        self.u_m = parameters[2]
        self.l_a = parameters[3]
        self.u_a = parameters[4]
        self.lamb = parameters[5]
        self.ce = torch.nn.CrossEntropyLoss()

    def calc_loss_G(self, x_norm):
        g = 1 / (self.u_a ** 2) * x_norm + 1 / x_norm
        return torch.mean(g)

    def forward(self, costheta, costhetam, labels, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        onehot = torch.zeros_like(costheta)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = onehot * costhetam + (1.0 - onehot) * costheta
        loss_ce = self.ce(logits, labels)

        loss = loss_ce + self.lamb * loss_g
        return loss


def get_loss(name, parameters_list: list):
    if name == 'ARCFACE':
        return ArcFace(parameters_list), 2
    elif name == 'COSFACE':
        return CosFace(parameters_list), 2
    elif name == 'ELASTICARC':
        return ElasticARC(parameters_list), 2
    elif name == 'ELASTICCOS':
        return ElasticCOS(parameters_list), 2
    elif name == 'SOFTMAX':
        return SOFTMAX(parameters_list), 0
    elif name == 'NORMFACE':
        return NORMFACE(parameters_list), 2
    elif name == 'X2SOFTMAX':
        return X2Softmax(parameters_list), 2
    elif name == 'MAGFACE':
        return MAGFACELOSS(parameters_list), 3
    elif name == 'DYNARCFACE':
        return DYNARCLOSS(parameters_list), 4

