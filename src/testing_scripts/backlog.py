def filter_hidden_states(pass_over_flag, iter_num, rna_seq, df_iter, feature_matrix,
                         matrix_iter, encoder_hidden_states):
    if pass_over_flag == 0:
        if iter_num * cfg.cut_seq_len <= rna_seq.loc[df_iter]["start"] - 1 < (iter_num + 1) * cfg.cut_seq_len:
            start = abs(rna_seq.loc[df_iter]["start"] - 1) % 100
            if rna_seq.loc[df_iter]["end"] - 1 < (iter_num + 1) * cfg.cut_seq_len:
                stop = abs(rna_seq.loc[df_iter]["end"] - 1) % 100
            else:
                stop = cfg.cut_seq_len
                pass_over_flag = 1

            for i in range(stop - start):
                feature_matrix.loc[matrix_iter + i, 0:3] = encoder_hidden_states[start + i,]
                feature_matrix.loc[matrix_iter + i, "label"] = rna_seq.loc[df_iter]["label"]

            df_iter += 1
            matrix_iter += (stop - start)
    else:
        start = 0
        if rna_seq.loc[df_iter]["stop"] - 1 < (iter_num + 1) * cfg.cut_seq_len:
            stop = rna_seq.loc[df_iter]["end"]
            pass_over_flag = 0
        else:
            stop = cfg.cut_seq_len

        for i in range(stop - start):
            feature_matrix.loc[matrix_iter + i, 0:3] = encoder_hidden_states[start + i,]
            feature_matrix.loc[matrix_iter + i, "label"] = rna_seq.loc[df_iter]["label"]

        df_iter += 1
        matrix_iter += (stop - start)

    return pass_over_flag, iter_num, df_iter, feature_matrix, matrix_iter


def run_tads(self, cfg):
    print("Running TADs")

    fire_ob = Fires()
    fire_ob.get_tad_data(self.fire_path, self.fire_cell_names)
    tad_filtered = fire_ob.filter_tad_data(self.chr_list_tad)
    mean_map_dict = {}
    cls_mode = 'ind'
    feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

    for col in range(7):
        tad_cell = tad_filtered[col]
        tad_cell['target'] = 1
        tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
        tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)
        tad_cell = fire_ob.augment_tad_negatives(cfg, tad_cell)

        mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(tad_cell)

        feature_matrix = self.Avo_downstream_helper_ob.filter_states(self.avocado_features, feature_matrix,
                                                                     mask_vector, label_ar)

        # feature_matrix = self.downstream_helper_ob.balance_classes(feature_matrix)

        mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

        mean_map_dict[self.fire_cell_names[col]] = mean_map

        print("cell name : {} - MAP : {}".format(self.fire_cell_names[col], mean_map))

    np.save(self.saved_model_dir + 'map_dict_tad.npy', mean_map_dict)

    return mean_map_dict


def run_tads(self, cfg):
    fire_ob = Fires()
    fire_ob.get_tad_data(self.fire_path, self.fire_cell_names)
    tad_filtered = fire_ob.filter_tad_data(self.chr_list_tad)
    mean_map_dict = {}
    cls_mode = 'ind'

    for col in range(7):
        tad_cell = tad_filtered[col]
        tad_cell['target'] = 1
        tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
        tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)
        tad_cell = fire_ob.augment_tad_negatives(cfg, tad_cell)
        mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(tad_cell)

        feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                      self.run_features_tad,
                                                                      self.feat_mat_tad,
                                                                      self.downstream_main)

        # feature_matrix = self.downstream_helper_ob.balance_classes(feature_matrix)

        mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

        mean_map_dict[self.fire_cell_names[col]] = mean_map

    np.save(self.saved_model_dir + 'map_dict_tad.npy', mean_map_dict)

    return mean_map_dict
