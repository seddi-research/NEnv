import bisect as bi

from NEnv.Utils.utils import clamp


class Distribution1D:
    def __init__(self, f):
        self.func = f

        n = len(f)

        # Compute CDF
        cdf = []
        cdf.append(0.0)

        for i in range(1, n + 1):
            cdf.append(cdf[i - 1] + f[i - 1] / n)

        # Transform step function integral into CDF
        self.funcInt = cdf[n]
        if (self.funcInt == 0):
            for i in range(1, n + 1):
                cdf[i] = float(i) / float(n)
        else:
            for i in range(1, n + 1):
                cdf[i] = cdf[i] / self.funcInt

        self.cdf = cdf

    def Count(self):
        return len(self.func)

    def SampleContinuous(self, u):
        # offset = bi.bisect_right(self.cdf, u) - 1

        offset = clamp(bi.bisect_right(self.cdf, u) - 1, 0, len(self.cdf) - 2)

        # Compute offset along CDF segment
        du = u - self.cdf[offset]
        diff = self.cdf[offset + 1] - self.cdf[offset]
        if diff > 0:
            du = du / diff

        pdf = self.func[offset] / self.funcInt if self.funcInt > 0 else 0
        # pdf = self.func[offset] / (self.funcInt * self.Count())  if self.funcInt > 0 else 0

        return (float(offset) + du) / self.Count(), pdf, offset

    def SampleContinuousValue(self, u):

        offset = clamp(bi.bisect_right(self.cdf, u) - 1, 0, len(self.cdf) - 2)

        # Compute offset along CDF segment
        du = u - self.cdf[offset]
        diff = self.cdf[offset + 1] - self.cdf[offset]
        if diff > 0:
            du = du / diff

        return (float(offset) + du) / self.Count()

    def SampleContinuousValueOffset(self, u):

        offset = clamp(bi.bisect_right(self.cdf, u) - 1, 0, len(self.cdf) - 2)

        # Compute offset along CDF segment
        du = u - self.cdf[offset]
        diff = self.cdf[offset + 1] - self.cdf[offset]
        if diff > 0:
            du = du / diff

        return (float(offset) + du) / self.Count(), offset

    def FuncInt(self):
        return self.funcInt

    def Func(self):
        return self.func

    def DiscretePDF(self, index):
        return self.func[index] / (self.funcInt * self.Count())


class Distribution2D:
    """
    This class can sample from a 2D distribution as described in Section 3.2 in https://diglib.eg.org/handle/10.1111/cgf14883
    """

    def __init__(self, data, nu, nv):
        self.m_funcInt = 0.0

        self.m_conditional = []

        for v in range(nv):

            subData = data[v * nu: v * nu + nu]

            self.m_conditional.append(Distribution1D(subData))

            partialFunc = 0.0

            for u in range(nu):
                partialFunc = partialFunc + data[v * nu + u]

            self.m_funcInt = self.m_funcInt + partialFunc / nu

        self.m_funcInt = self.m_funcInt / nv

        marginalFunc = []
        for v in range(nv):
            marginalFunc.append(self.m_conditional[v].FuncInt())

        self.m_marginal = Distribution1D(marginalFunc)

    def Sample(self, u):

        d1, pdf1, v = self.m_marginal.SampleContinuous(u[1])

        d0, pdf0, _ = self.m_conditional[v].SampleContinuous(u[0])

        return [d0, d1], pdf0 * pdf1

    def SampleValues(self, u):

        d1, v = self.m_marginal.SampleContinuousValueOffset(u[1])

        d0 = self.m_conditional[v].SampleContinuousValue(u[0])

        return [d0, d1]

    def PDF(self, p):
        iu = clamp(int(p[0] * self.m_conditional[0].Count()), 0, int(self.m_conditional[0].Count()) - 1)
        iv = clamp(int(p[1] * self.m_marginal.Count()), 0, int(self.m_marginal.Count()) - 1);
        return self.m_conditional[iv].Func()[iu] / self.m_marginal.FuncInt()
