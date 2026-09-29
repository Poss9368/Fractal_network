"""Conservación del objetivo, convergencia y escalabilidad de FIRE."""
import unittest
import numpy as np
from scipy.optimize import minimize
from .dynamic import (fire_minimize, maximum_length_angle,
                      rest_deformations_from_thetas, modified_hamiltonian,
                      modified_hamiltonian_gradient, constraint_term_gradient)
from .fire import _evaluate, _length_gradient


class FIRETests(unittest.TestCase):
    def case(self,n):
        rest=np.full(n,.5*maximum_length_angle(n))
        return rest,np.ones(n),np.ones((n,2)),rest_deformations_from_thetas(rest)

    def test_transformed_energy_and_gradients(self):
        rng=np.random.default_rng(8)
        theta=rng.normal(.1,.02,9)
        rest=rng.normal(0,.03,(9,2))
        weights=rng.uniform(.2,1.5,9)
        fam=rng.uniform(0,1,(9,2))
        z=rng.normal(0,.02,9)
        prefix=np.r_[0,np.cumsum(theta[:-1])]
        residual=np.column_stack((theta-prefix-rest[:,0],theta+prefix-rest[:,1]))
        displaced=theta+np.diff(np.r_[0,z])
        energy,grad,gm,gr=_evaluate(z,theta,residual,weights,fam,1.3)
        actual=modified_hamiltonian_gradient(displaced,weights,fam,rest,1.3)
        self.assertAlmostEqual(energy,modified_hamiltonian(displaced,weights,fam,rest,1.3),places=12)
        np.testing.assert_allclose(np.cumsum(grad[::-1])[::-1],actual,atol=1e-12)
        self.assertAlmostEqual(gm,np.max(abs(actual)),places=12)
        self.assertAlmostEqual(gr,np.sqrt(np.mean(actual**2)),places=12)
        for i in range(9):
            step=np.eye(9)[i]*1e-6
            numeric=(_evaluate(z+step,theta,residual,weights,fam,1.3)[0]-_evaluate(z-step,theta,residual,weights,fam,1.3)[0])/2e-6
            self.assertAlmostEqual(grad[i],numeric,places=7)

    def test_length_gradient_at_cosine_zero(self):
        theta=np.array([np.pi,.2,.3])
        value,grad=_length_gradient(theta)
        self.assertTrue(np.isfinite(value))
        for i in range(3):
            step=np.eye(3)[i]*1e-6
            numeric=(_length_gradient(theta+step)[0]-_length_gradient(theta-step)[0])/2e-6
            self.assertAlmostEqual(grad[i],numeric,places=8)

    def test_matches_independent_minimizer(self):
        theta,w,fam,rest=self.case(12)
        fam[:,0]=np.linspace(.3,1.,12)
        fam[:,1]=np.linspace(1.,.4,12)
        w[:]=np.linspace(.4,1.6,12)
        for force in (.01,1.,100.):
            result=fire_minimize(theta,w,fam,rest,force,return_info=True,ftol=1e-9)
            other=minimize(modified_hamiltonian,theta,args=(w,fam,rest,force),
                jac=modified_hamiltonian_gradient,method='BFGS',options={'gtol':1e-9})
            self.assertTrue(result.converged)
            np.testing.assert_allclose(result.thetas,other.x,atol=2e-7)
            self.assertAlmostEqual(result.energy,other.fun,places=9)

    def test_inputs_unchanged_and_zero_load(self):
        args=self.case(16)
        before=[x.copy() for x in args]
        result=fire_minimize(*args,0.,return_info=True)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations,0)
        fire_minimize(*args,1.)
        for a,b in zip(args,before): np.testing.assert_array_equal(a,b)

    def test_failure_is_explicit(self):
        args=self.case(4)
        with self.assertRaises(RuntimeError): fire_minimize(*args,1.,max_steps=0)
        r=fire_minimize(*args,1.,max_steps=0,return_info=True,raise_on_failure=False)
        self.assertFalse(r.converged)
        self.assertEqual(r.iterations,0)

    def test_invalid_parameters(self):
        args=self.case(4)
        for kwargs in ({'dt':0},{'dt':2,'dt_max':1},{'ftol':0},{'n_min':1.5},{'max_steps':-1},{'fdec':1},{'rtol':-1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                fire_minimize(*args,1.,**kwargs)
        bad=args[0].copy();bad[0]=np.nan
        with self.assertRaises(ValueError): fire_minimize(bad,*args[1:],1.)

    def test_large_systems(self):
        for n in (4096,16384):
            args=list(self.case(n))
            for load in (1e-5,1.,1e3,1e5):
                result=fire_minimize(*args,load,return_info=True,max_steps=3000)
                self.assertLessEqual(result.gradient_max,1e-7)
                self.assertLess(result.iterations,3000)
                self.assertTrue(np.all(np.isfinite(result.thetas)))
                args[0]=result.thetas

    def test_without_preconditioner(self):
        args=self.case(8)
        r=fire_minimize(*args,.3,precondition=False,dt=.01,dt_max=.1,return_info=True)
        self.assertTrue(r.converged)
        self.assertLess(np.max(abs(modified_hamiltonian_gradient(r.thetas,*args[1:],.3))),1.1e-7)


if __name__=='__main__': unittest.main()
