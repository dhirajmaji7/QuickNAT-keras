# The actual paper uses a combination of dice loss and weighted crossentropy, but instead we have implemented a generalised dice loss
# with weighted crossentropy.
# generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
 
# assign the weights here in proper order
weights=np.array([1,5,2,4])
weights = K.variable(weights)
# weights are assigned in this order : normal,necrotic,edema,enhancing tumor

# weighted crossentropy
def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) 
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

# computes the sum of two losses : generalised dice loss and weighted cross entropy
def gen_dice_loss(y_true, y_pred):

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL+weighted_log_loss(y_true,y_pred)
