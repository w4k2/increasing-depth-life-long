import copy
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger
from avalanche.models import avalanche_forward
import torch.nn.functional as F
import torch

class MIR(BaseStrategy):

    def __init__(self, model, optimizer, criterion,
                 patterns_per_exp: int,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None, evaluator=default_logger, eval_every=-1):
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

        self.patterns_per_experience = int(patterns_per_exp)

        self.buffers = []  # one AvalancheDataset for each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None
    
    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            
            y_hat = avalanche_forward(strategy.model, strategy.mb_x, strategy.mb_task_id)
            pre_loss = F.cross_entropy(y_hat, strategy.mb_y, reduction='none')

            strategy.optimizer.zero_grad()
            pre_loss.backward()


            # bx, by, bt, subsample = buffer.sample(args.subsample, exclude_task=task, ret_ind=True)
            mb = self.sample_from_memory()
            bx, by, bt = mb[0], mb[1], mb[-1]
            
            grad_dims = []
            for param in strategy.model.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = self.get_grad_vector(strategy.device, strategy.model.parameters, grad_dims)
            model_temp = self.get_future_step_parameters(strategy.model, grad_vector, grad_dims, lr=strategy.optimizer.lr)

            with torch.no_grad():
                logits_track_pre = strategy.model(bx)
                buffer_hid = model_temp.return_hidden(bx)
                logits_track_post = model_temp.linear(buffer_hid)

                post_loss = F.cross_entropy(logits_track_post, by , reduction="none")

                scores = post_loss - pre_loss
                big_ind = scores.sort(descending=True)[1][:strategy.train_mb_size//2]

                idx = subsample[big_ind]

            mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]

            self.optimizer.zero_grad()


            # mb = self.sample_from_memory()
            # xref, yref, tid = mb[0], mb[1], mb[-1]
            # xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            # out = avalanche_forward(strategy.model, xref, tid)
            # loss = strategy._criterion(out, yref)
            # loss.backward()
            # # gradient can be None for some head on multi-headed models
            # self.reference_gradients = [
            #     p.grad.view(-1) if p.grad is not None
            #     else torch.zeros(p.numel(), device=strategy.device)
            #     for n, p in strategy.model.named_parameters()]
            # self.reference_gradients = torch.cat(self.reference_gradients)
            # strategy.optimizer.zero_grad()


    def get_grad_vector(self, device, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims))
        if device.startswith('cuda'): 
            grads = grads.to(device)

        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads


    def get_future_step_parameters(self, this_net, grad_vector, grad_dims, lr=1):
        new_net=copy.deepcopy(this_net)
        self.overwrite_grad(new_net.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data=param.data - lr*param.grad.data
        return new_net


    def overwrite_grad(self, parameters, new_grad, grad_dims):
        cnt = 0
        for param in parameters():
            param.grad=torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1


def retrieve_replay_update(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    """ finds buffer samples with maxium interference """
    updated_inds = None

    hid = model.return_hidden(input_x)

    logits = model.linear(hid)
    if args.multiple_heads:
        logits = logits.masked_fill(loader.dataset.mask == 0, -1e9)

    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    bx, by, bt, subsample = buffer.sample(args.subsample, exclude_task=task, ret_ind=True)
    grad_dims = []
    for param in model.parameters():
        grad_dims.append(param.data.numel())
    grad_vector = get_grad_vector(args, model.parameters, grad_dims)
    model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

    with torch.no_grad():
        logits_track_pre = model(bx)
        buffer_hid = model_temp.return_hidden(bx)
        logits_track_post = model_temp.linear(buffer_hid)

        if args.multiple_heads:
            mask = torch.zeros_like(logits_track_post)
            mask.scatter_(1, loader.dataset.task_ids[bt], 1)
            assert mask.nelement() // mask.sum() == args.n_tasks
            logits_track_post = logits_track_post.masked_fill(mask == 0, -1e9)
            logits_track_pre = logits_track_pre.masked_fill(mask == 0, -1e9)

        pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
        post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
        scores = post_loss - pre_loss
        EN_logits = entropy_fn(logits_track_pre)
        if args.compare_to_old_logits:
            old_loss = F.cross_entropy(buffer.logits[subsample], by,reduction="none")

            updated_mask = pre_loss < old_loss
            updated_inds = updated_mask.data.nonzero().squeeze(1)
            scores = post_loss - torch.min(pre_loss, old_loss)

        all_logits = scores
        big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]

        idx = subsample[big_ind]

    mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]

    logits_buffer = model(mem_x)
    if args.multiple_heads:
        mask = torch.zeros_like(logits_buffer)
        mask.scatter_(1, loader.dataset.task_ids[b_task_ids], 1)
        assert mask.nelement() // mask.sum() == args.n_tasks
        logits_buffer = logits_buffer.masked_fill(mask == 0, -1e9)
    F.cross_entropy(logits_buffer, mem_y).backward()

    if updated_inds is not None:
        buffer.logits[subsample[updated_inds]] = deepcopy(logits_track_pre[updated_inds])
    opt.step()
    return model
