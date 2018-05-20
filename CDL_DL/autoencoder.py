from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import model
import logging
import mxnet as mx
from mxnet import misc


class EncoderProto(model.MXModel):

	def setup(self, dimension , penalty_sparse=None, dropout_pt=None, dropout_ft=None, in_act=None, inter_act='relu', o_act=None):
        
        self.dimension = dimension
        self.dropout_ft = dropout_ft
        self.dropout_pt = dropout_pt
        self.inter_act = inter_act
        self.in_act = in_act
        self.o_act = o_act
        self.N = len(dimension) - 1
        self.stack = []

        self.V = mx.symbol.Variable('V')
        self.lambda_v_rt = mx.symbol.Variable('lambda_v_rt')
        self.data = mx.symbol.Variable('data')
        
        
        for i in range(self.N):
            if i == 0:
                d_act = in_act
                idrop = None
            else:
                d_act = inter_act
                idrop = dropout_pt
            if i == self.N-1:
                e_act = o_act
                odrop = None
            else:
                e_act = inter_act
                odrop = dropout_pt


            stack_i, args_i, grad_iargs, mult_iargs, aux_i = self.stack_make(i, self.data, dimension[i], dimension[i+1],
                                                penalty_sparse, idrop, odrop, e_act, d_act)
            self.stack.append(stack_i)
            self.args.update(args_i)
            self.args_grad.update(grad_iargs)
            self.args_mult.update(mult_iargs)
            self.auxs.update(aux_i)
        self.enc, self.internals = self.make_encoder(self.data, dimension, penalty_sparse, ft_dropout, inter_act, o_act)
        self.dec = self.make_decoder(self.encoder, dimension, penalty_sparse, ft_dropout, inter_act, in_act)


        if in_act == 'softmax':
            self.valueLoss = self.decoder
        else:
            loss_fe = mx.symbol.LinearRegressionOutput(data=self.lambda_v_rt*self.enc,
                label=self.lambda_v_rt*self.V)
            loss_fr = mx.symbol.LinearRegressionOutput(data=self.dec, label=self.data)
            self.valueLoss = mx.symbol.Group([loss_fe, loss_fr])

    def stack_make(self, sti, datapoints, total_input, total_hidden, penalty_sparse=None, idrop=None,
                   odropout=None, encoder_act='relu', decoder_act='relu'):
        val = data
        # if value is dropout
        if idrop:
            val = mx.symbol.Dropout(data=val, p=idrop)
        val = mx.symbol.FullyConnected(name='encoder_%d'%sti, data=val, total_hidden=total_hidden)
        # if encoder is activated
        if encoder_act:
            val = mx.symbol.Activation(data=val, act_type=encoder_act)
            if encoder_act == 'sigmoid' and penalty_sparse:
                val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='sparse_encoder_%d' % sti, penalty=penalty_sparse)
        # if there is a dropout
        if odropout:
            val = mx.symbol.Dropout(data=val, p=odropout)
        val = mx.symbol.FullyConnected(name='decoder_%d'%sti, data=val, total_hidden=total_input)
        # if decoder act is softmaval
        if decoder_act == 'softmaval':
            val = mx.symbol.Softmax(data=val, label=datapoints, prob_label=True, act_type=decoder_act)
        # if decoder act is sigmoid
        elif decoder_act:
            val = mx.symbol.Activation(data=val, act_type=decoder_act)
            if decoder_act == 'sigmoid' and penalty_sparse:
                val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='sparse_decoder_%d' % sti, penalty=penalty_sparse)
            val = mx.symbol.LinearRegressionOutput(data=val, label=datapoints)
        else:
            val = mx.symbol.LinearRegressionOutput(data=val, label=data)

        args = {'%d_weight'%sti: mx.nd.empty((total_hidden, total_input), self.xpu),
                '%d_bias'%sti: mx.nd.empty((total_hidden,), self.xpu),
                '%d_weight'%sti: mx.nd.empty((total_input, total_hidden), self.xpu),
                '_%d_bias'%sti: mx.nd.empty((total_input,), self.xpu),}
        args_grad = {'%d_weight'%sti: mx.nd.empty((total_hidden, total_input), self.xpu),
                     '%d_bias'%sti: mx.nd.empty((total_hidden,), self.xpu),
                     '%d_weight'%sti: mx.nd.empty((total_input, total_hidden), self.xpu),
                     '%d_bias'%sti: mx.nd.empty((total_input,), self.xpu),}
        args_mult = {'%d_weight'%sti: 1.0,
                     '%d_bias'%sti: 2.0,
                     '%d_weight'%sti: 1.0,
                     '%d_bias'%sti: 2.0,}
        auxs = {}
        if decoder_act == 'sigmoid' and penalty_sparse:
            auxs['%d_moving_avg' % sti] = mx.nd.ones((total_input), self.xpu) * 0.5
        if encoder_act == 'sigmoid' and penalty_sparse:
            auxs['%d_moving_avg' % sti] = mx.nd.ones((total_hidden), self.xpu) * 0.5 
        init = mx.initializer.Uniform(0.07)
        for k,v in args.items():
            init(k,v)

        return val, args, args_grad, args_mult, auxs

    def make_encoder(self, data, dimensions, penalty_sparse=None, dropout=None, i_act='relu', o_act=None):
        val = data
        internals = []
        N = len(dimensions) - 1
        for i in range(N):
            val = mx.symbol.FullyConnected(name='encoder_%d'%i, data=val, total_hidden=dimensions[i+1])
            if i_act and i < N-1:
                val = mx.symbol.Activation(data=val, act_type=i_act)
                #if the function is sigmoid and penalty is sparse
                if i_act=='sigmoid' and penalty_sparse:
                    val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='%d' % i, penalty=penalty_sparse)
            elif o_act and i == N-1:
                val = mx.symbol.Activation(data=val, act_type=o_act)
                #if the penalty is sigmoid and the penalty is sparse
                if o_act=='sigmoid' and penalty_sparse:
                    val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='%d' % i, penalty=penalty_sparse)
            if dropout:
                val = mx.symbol.Dropout(data=val, p=dropout)
            internals.append(val)
        return val, internals

    def make_decoder(self, totalfeatures, dimensions, penalty_sparse=None, dropout=None, i_act='relu', in_act=None):
        val = totalfeatures
        N = len(dimensions) - 1
        #for all i values
        for i in reversed(range(N)):
            val = mx.symbol.FullyConnected(name='decoder_%d'%i, data=val, total_hidden=dimensions[i])
            if i_act and i > 0:
                val = mx.symbol.Activation(data=val, act_type=i_act)
                # if the function is sigmoid and the penalty is sparse
                if i_act=='sigmoid' and penalty_sparse:
                    val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='%d' % i, penalty=penalty_sparse)
            elif in_act and i == 0:
                val = mx.symbol.Activation(data=val, act_type=in_act)
                # if the function is sigmoid and the penalty is sparse
                if in_act=='sigmoid' and penalty_sparse:
                    val = mx.symbol.IdentityAttachKLSparseReg(data=val, name='%d' % i, penalty=penalty_sparse)
            if dropout and i > 0:
                val = mx.symbol.Dropout(data=val, p=dropout)
        return val

    def layerwise_pretrain(self, X, b_size, total_iter, opti, l_rate, decay, scheduler_lr=None):
        def l2_norm(label, pred):
            return np.mean(np.square(label-pred))/2.0
        solver = Solver(opti, momentum=0.9, wd=decay, learning_rate=l_rate, scheduler_lr=scheduler_lr)
        # procedding solver.set_metric
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        # procedding solver.set_monitor
        solver.set_monitor(Monitor(1000))
        # procedding solver. data_iter
        data_iter = mx.io.NDArrayIter({'data': X}, b_size=b_size, shuffle=True,
                                      last_batch_handle='roll_over')
        # processing all is in range
        for i in range(self.N):
            if i == 0:
                data_iter_i = data_iter
            else:
                X_i = model.extract_feature(self.internals[i-1], self.args, self.auxs,
                                            data_iter, X.shape[0], self.xpu).values()[0]
                data_iter_i = mx.io.NDArrayIter({'data': X_i}, b_size=b_size,
                                                last_batch_handle='roll_over')
            logging.info('Pre-training layer %d...'%i)
            solver.solve(self.xpu, self.stacks[i], self.args, self.args_grad, self.auxs, data_iter_i,
                         0, total_iter, {}, False)

    def finetune(self, X, R, V, lambda_v_rt, lambda_u, lambda_v, dir_save, b_size, total_iter, opti, l_rate, decay, scheduler_lr=None):
        def l2_norm(label, pred):
            print(type(label))
            print(type(pred))
            print(np.shape(label))
            print(np.shape(pred))
            return np.mean(np.square(label-pred))/2.0
        solver = Solver(opti, momentum=0.9, wd=decay, learning_rate=l_rate, scheduler_lr=scheduler_lr)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            lambda_v_rt},
                b_size=b_size, shuffle=False,
                last_batch_handle='pad')
        logging.info('Fine tuning...')
        # self.loss is the net
        U, V, theta, BCD_loss = solver.solve(X, R, V, lambda_v_rt, lambda_u,
            lambda_v, dir_save, b_size, self.xpu, self.loss, self.args, self.args_grad, self.auxs, data_iter,
            0, total_iter, {}, False)
        return U, V, theta, BCD_loss

    # modified by hog
    def eval(self, X, V, lambda_v_rt):
        b_size = 100
        data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            lambda_v_rt},
            b_size=b_size, shuffle=False,
            last_batch_handle='pad')
        # modified by hog
        Y = model.extract_feature(self.loss[1], self.args, self.auxs, data_iter,
                                 X.shape[0], self.xpu).values()[0]
        return np.sum(np.square(Y-X))/2.
