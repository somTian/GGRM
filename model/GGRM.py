import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import l2_loss, inner_product, log_loss
from data import PairwiseSampler, PointwiseSampler
from time import time
import pandas as pd
from scipy.sparse import csr_matrix


class GGRM(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(GGRM, self).__init__(dataset, config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_dim = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        # self.n_layers = config['n_layers']
        self.n_depth = config['n_layers']
        self.verbose = config['verbose']

        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.all_users = list(self.user_pos_train.keys())


        ui_adj,ori_ui_adj = self.create_ui_adj_mat()
        self.norm_ui_adj = self.norm_adj_mat(ui_adj,config['adj_type'])

        self.n_groups, self.n_links, self.group_matrix = self._load_group()
        ug_adj,ori_ug_adj = self.create_u_g_adj_mat()
        ori_gu_adj = ori_ug_adj.T

        self.norm_ug_adj = self.norm_adj_mat(ug_adj,config['adj_type'])

        print('start create gi matric')
        self.ori_gi_adj = ori_gu_adj.dot(ori_ui_adj)

        gi_adj = self.create_gi_adj_mat()

        self.norm_gi_adj = self.norm_adj_mat(gi_adj, config['adj_type'])

        self.sess = sess


    def getgroupMatrix(self):
        user_group_indices_list = []
        user_group_values_list = []
        dok_matrix = self.group_matrix.todok()

        for (user, group), value in dok_matrix.items():
            user_group_indices_list.append([user, group])
            user_group_values_list.append(1.0/len(dok_matrix[user]))

        self.user_group_indices_list = np.array(user_group_indices_list).astype(np.int32)
        self.user_group_values_list = np.array(user_group_values_list).astype(np.float32)

        self.user_group_dense_shape = np.array([self.n_users, self.n_groups]).astype(np.int32)

        user_group_sparse_matrix = tf.SparseTensor(indices=self.user_group_indices_list,
                                                            values=self.user_group_values_list,
                                                            dense_shape=self.user_group_dense_shape)

        return user_group_sparse_matrix


    def _load_group(self):
        group_file = "user_group.dat"
        group_data = pd.read_csv(group_file, sep='\t', header=None, names=['user','group'])

        unique_group = group_data["group"].unique()
        self.groupids = pd.Series(data=range(len(unique_group)), index=unique_group).to_dict()
        group_data["group"] = group_data["group"].map(self.groupids)
        group_data["user"] = group_data["user"].map(self.dataset.userids)
        num_groups =  max(group_data["group"]) + 1


        group_links = [1.0] * len(group_data["user"])
        num_links = len(group_links)
        group_matrix = csr_matrix((group_links, (group_data["user"], group_data["group"])),
                                       shape=(self.n_users, num_groups))

        print("Group num:",num_groups)
        print("Group link:",num_links)

        return num_groups,num_links,group_matrix


    def get_group_interactions(self):
        dok_matrix = self.group_matrix.todok()
        users_list, groups_list = [], []
        for (user, group), value in dok_matrix.items():
            users_list.append(user)
            groups_list.append(group)

        return users_list, groups_list

    def get_gi_interactions(self):

        dok_matrix = self.ori_gi_adj.todok()
        groups_list, items_list = [], []
        for (group, item), value in dok_matrix.items():
            groups_list.append(group)
            items_list.append(item)

        return groups_list, items_list


    @timer
    def create_u_g_adj_mat(self):
        user_list, group_list = self.get_group_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        group_np = np.array(group_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_groups
        tmp_adj = sp.csr_matrix((ratings, (user_np, group_np+self.n_users)), shape=(n_nodes, n_nodes))
        ori_adj = sp.csr_matrix((ratings, (user_np, group_np)), shape=(self.n_users, self.n_groups))
        return tmp_adj,ori_adj

    @timer
    def create_ui_adj_mat(self):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        ori_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.n_users, self.n_items))
        return tmp_adj,ori_adj

    @timer
    def create_gi_adj_mat(self):
        group_list, item_list = self.get_gi_interactions()
        group_np = np.array(group_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(group_np, dtype=np.float32)
        n_nodes = self.n_groups + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (group_np, item_np + self.n_groups)), shape=(n_nodes, n_nodes))
        return tmp_adj

    def norm_adj_mat(self,tmp_adj,adj_type):
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix


    def _create_variable(self):

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        # self.label = tf.placeholder(tf.float32, shape=(None,))


        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        self.weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        self.weights['group_embedding'] = tf.Variable(initializer([self.n_groups, self.emb_dim]), name='group_embedding')


    def build_graph(self):
        self._create_variable()
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
        self.gi_embeddings, self.ig_embeddings = self._create_gigcn_embed()
        self.ug_embeddings, self.gu_embeddings = self._create_groupgcn_embed()


        self.uf_embeddings = tf.concat([self.ua_embeddings, self.ug_embeddings], axis=1)
        self.if_embeddings = tf.concat([self.ia_embeddings, self.ig_embeddings], axis=1)



        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.uf_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.if_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.if_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, 2 * self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, 2 * self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.uf_embeddings),
                           tf.assign(self.item_embeddings_final, self.if_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.users)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, 2 * self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, 2 * self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.uf_embeddings),
                           tf.assign(self.item_embeddings_final, self.if_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.users)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                           self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)

        # self.mf_loss, self.emb_loss = self.creater_point_loss(self.u_g_embeddings,self.pos_i_g_embeddings,self.label)

        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_lightgcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_ui_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_depth):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_g_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_groupgcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_ug_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.group_embedding], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_depth):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_g_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_groups], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gigcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_gi_adj)

        ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_depth):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_gi_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = inner_product(users, pos_items)
        neg_scores = inner_product(users, neg_items)

        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre, self.neg_i_g_embeddings_pre,self.weights['group_embedding'])

        mf_loss = tf.reduce_sum(log_loss(pos_scores - neg_scores))

        emb_loss = self.reg * regularizer

        return mf_loss, emb_loss



    def train_model(self):

        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)

        print(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(data_iter)
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed_dict = {self.users: bat_users,
                        self.pos_items: bat_pos_items,
                        self.neg_items: bat_neg_items}
                # self.sess.run(self.opt, feed_dict=feed)
                loss, _ = self.sess.run((self.loss, self.opt), feed_dict=feed_dict)
                total_loss += loss

            print("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
                                                       time() - training_start_time))
            if epoch % self.verbose == 0:
                all_user_result, result = self.evaluate()
                print("epoch %d:\t%s" % (epoch, result))


    @timer
    def evaluate(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, users, candidate_items=None):
        feed_dict = {self.users: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        if candidate_items is not None:
            ratings = [ratings[idx][u_item] for idx, u_item in enumerate(candidate_items)]
        return ratings
