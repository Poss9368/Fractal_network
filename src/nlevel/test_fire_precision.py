"""Precisión del estado hi+lo y regresión del barrido completo a 1e-12."""
import unittest
from decimal import Decimal as D, localcontext
import numpy as np
from .dynamic import fire_minimize, maximum_length_angle, rest_deformations_from_thetas
from .fire_precision import evaluate


def decimal_gradient(theta, correction, weights, families, rest, load):
    """Referencia independiente, 55 dígitos, ángulos pequeños de estas pruebas."""
    with localcontext() as context:
        context.prec=55
        total, product, prefix = D(1), D(1), D(0)
        direct, future, tangent = [], [], []
        for i in range(len(theta)):
            angle=D(float(theta[i]))+D(float(correction[i]))
            x=angle/2
            sine, cosine, st, ct = x,D(1),x,D(1)
            for k in range(1,20):
                st *= -x*x/D(2*k*(2*k+1))
                ct *= -x*x/D((2*k-1)*2*k)
                sine+=st;cosine+=ct
            t=sine/cosine
            tangent.append(t);total+=t;product*=cosine
            rm=angle-prefix-D(float(rest[i,0]))
            rp=angle+prefix-D(float(rest[i,1]))
            am=D(float(weights[i]))*D(float(families[i,0]))
            ap=D(float(weights[i]))*D(float(families[i,1]))
            direct.append(2*(am*rm+ap*rp))
            future.append(2*(-am*rm+ap*rp))
            prefix+=angle
        suffix=D(0)
        gradient=np.empty(len(theta))
        for i in range(len(theta)-1,-1,-1):
            gx=product/2*(1+tangent[i]**2-tangent[i]*total)
            gradient[i]=float(direct[i]+suffix-D(float(load))*gx)
            suffix+=future[i]
        return gradient


class PreciseFIRETests(unittest.TestCase):
    def case(self,n):
        theta=np.full(n,.5*maximum_length_angle(n))
        return theta,np.ones(n),np.ones((n,2)),rest_deformations_from_thetas(theta)

    def test_correction_cannot_be_silently_discarded(self):
        with self.assertRaisesRegex(ValueError,'return_info=True'):
            fire_minimize(*self.case(8),1.,ftol=1e-12)

    def test_extended_gradient_against_decimal(self):
        rng=np.random.default_rng(712)
        theta,weights,families,rest=self.case(16)
        theta += rng.normal(0,.01,len(theta))
        correction=rng.normal(0,1e-18,len(theta))
        weights[:]=rng.uniform(.2,1.5,len(theta))
        families[:]=rng.uniform(0,1,families.shape)
        result=evaluate(theta,correction,weights,families,rest,1e6)
        expected=decimal_gradient(theta,correction,weights,families,rest,1e6)
        np.testing.assert_allclose(np.cumsum(result[1][::-1])[::-1],expected,rtol=1e-14,atol=1e-10)
        self.assertAlmostEqual(result[2],np.max(np.abs(expected)),places=9)

    def test_complete_force_sweep_at_1e12(self):
        for n in (4096,16384):
            theta,weights,families,rest=self.case(n)
            correction=np.zeros(n)
            for force in np.r_[0.,np.logspace(-5,6,34)]:
                r=fire_minimize(theta,weights,families,rest,force,
                    ftol=1e-12,rtol=0.,theta_correction=correction,
                    return_info=True,max_steps=3000)
                theta,correction=r.thetas,r.theta_correction
                actual=evaluate(theta,correction,weights,families,rest,force)
                with self.subTest(n=n,force=force):
                    self.assertTrue(r.converged)
                    self.assertEqual(r.tolerance,1e-12)
                    self.assertLessEqual(actual[2],1e-12)
                    self.assertEqual(actual[2],r.gradient_max)
                    self.assertLessEqual(r.iterations,3000)
            # La prueba más rígida se valida sin reutilizar la aritmética DD.
            exact=decimal_gradient(theta,correction,weights,families,rest,1e6)
            self.assertLessEqual(np.max(np.abs(exact)),1e-12)
            self.assertLess(abs(np.max(np.abs(exact))-r.gradient_max),1e-20)
            self.assertGreater(r.rounded_gradient_max,r.tolerance)
            self.assertTrue(np.any(correction != 0))

    def test_budget_and_domain_failure(self):
        args=self.case(8)
        r=fire_minimize(*args,1.,ftol=1e-12,max_steps=0,return_info=True,raise_on_failure=False)
        self.assertFalse(r.converged)
        self.assertEqual(r.iterations,0)
        theta=np.array([1.6])
        with self.assertRaisesRegex(RuntimeError,'1.5 rad'):
            fire_minimize(theta,np.ones(1),np.ones((1,2)),
                          rest_deformations_from_thetas(theta),0.,ftol=1e-12,return_info=True)


if __name__=='__main__':unittest.main()
