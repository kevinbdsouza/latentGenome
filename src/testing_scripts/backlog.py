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
