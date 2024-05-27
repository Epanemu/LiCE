import numpy as np
import pandas as pd
import pytest

from LiCE.data.Features import Binary


class TestBinary:
    np_input = Binary(
        np.array(
            ["pos", "neg", "pos", "pos"],
        ),
        ["neg", "pos"],
        name="test",
    )
    pd_input = Binary(
        pd.Series(["pos", "neg", "pos", "pos"], name="test"),
        ["neg", "pos"],
    )

    pos = np.array(["pos"])
    neg = np.array(["neg"])
    pos_pd = pd.Series(["pos"])
    neg_pd = pd.Series(["neg"])

    longer = np.array(["pos", "neg", "pos", "pos", "neg", "pos", "neg"])
    longer_pd = pd.Series(["pos", "neg", "pos", "pos", "neg", "pos", "neg"])

    @pytest.fixture()
    def single_enc(self):
        return {
            "np": {
                "pos_res": self.np_input.encode(
                    self.pos, normalize=True, one_hot=False
                ),
                "neg_res": self.np_input.encode(
                    self.neg, normalize=True, one_hot=False
                ),
                "pos_res_ohe": self.np_input.encode(
                    self.pos_pd, normalize=True, one_hot=True
                ),
                "neg_res_ohe": self.np_input.encode(
                    self.neg_pd, normalize=True, one_hot=True
                ),
            },
            "pd": {
                "pos_res": self.pd_input.encode(
                    self.pos_pd, normalize=True, one_hot=False
                ),
                "neg_res": self.pd_input.encode(
                    self.neg_pd, normalize=True, one_hot=False
                ),
                "pos_res_ohe": self.pd_input.encode(
                    self.pos, normalize=True, one_hot=True
                ),
                "neg_res_ohe": self.pd_input.encode(
                    self.neg, normalize=True, one_hot=True
                ),
            },
        }

    @pytest.fixture()
    def single_dec(self, single_enc):
        return {
            "np": {
                "pos_res": self.np_input.decode(
                    single_enc["np"]["pos_res"], denormalize=True, return_series=False
                ),
                "neg_res": self.pd_input.decode(
                    single_enc["pd"]["neg_res"], denormalize=True, return_series=False
                ),
                "pos_res_ohe": self.np_input.decode(
                    single_enc["np"]["pos_res_ohe"],
                    denormalize=True,
                    return_series=False,
                ),
                "neg_res_ohe": self.pd_input.decode(
                    single_enc["pd"]["neg_res_ohe"],
                    denormalize=True,
                    return_series=False,
                ),
            },
            "pd": {
                "pos_res": self.pd_input.decode(
                    single_enc["pd"]["pos_res"], denormalize=True, return_series=True
                ),
                "neg_res": self.np_input.decode(
                    single_enc["np"]["neg_res"], denormalize=True, return_series=True
                ),
                "pos_res_ohe": self.pd_input.decode(
                    single_enc["pd"]["pos_res_ohe"],
                    denormalize=True,
                    return_series=True,
                ),
                "neg_res_ohe": self.np_input.decode(
                    single_enc["np"]["neg_res_ohe"],
                    denormalize=True,
                    return_series=True,
                ),
            },
        }

    @pytest.fixture()
    def longer_res(self):
        enc = {
            "np": self.np_input.encode(self.longer, normalize=True, one_hot=False),
            "np_ohe": self.np_input.encode(self.longer, normalize=True, one_hot=True),
            "pd": self.pd_input.encode(self.longer_pd, normalize=True, one_hot=False),
            "pd_ohe": self.pd_input.encode(
                self.longer_pd, normalize=True, one_hot=True
            ),
        }
        dec_np = {
            "np": self.np_input.decode(
                enc["np"], denormalize=True, return_series=False
            ),
            "np_ohe": self.pd_input.decode(
                enc["np_ohe"], denormalize=True, return_series=False
            ),
            "pd": self.pd_input.decode(
                enc["pd"], denormalize=True, return_series=False
            ),
            "pd_ohe": self.np_input.decode(
                enc["pd_ohe"], denormalize=True, return_series=False
            ),
        }
        dec_pd = {
            "np": self.np_input.decode(enc["np"], denormalize=True, return_series=True),
            "np_ohe": self.pd_input.decode(
                enc["np_ohe"], denormalize=True, return_series=True
            ),
            "pd": self.pd_input.decode(enc["pd"], denormalize=True, return_series=True),
            "pd_ohe": self.np_input.decode(
                enc["pd_ohe"], denormalize=True, return_series=True
            ),
        }
        return {"enc": enc, "dec_np": dec_np, "dec_pd": dec_pd}

    def test_binary_types(self, single_enc, single_dec):
        for enc in single_enc.values():
            for res in enc.values():
                assert isinstance(res, np.ndarray)
                assert res.dtype == np.float64
        for res in single_dec["np"].values():
            assert isinstance(res, np.ndarray)
        for res in single_dec["pd"].values():
            assert isinstance(res, pd.Series)

    def test_binary_is_correct(self, single_enc):
        for enc in single_enc.values():
            assert all(enc["pos_res"] == 1)
            assert all(enc["neg_res"] == 0)
            assert np.all(enc["neg_res_ohe"] == np.array([1, 0]))
            assert np.all(enc["pos_res_ohe"] == np.array([0, 1]))
        for key in single_enc["np"].keys():
            assert np.all(single_enc["np"][key] == single_enc["pd"][key])

    def test_binary_keeps_shape(self, single_enc, single_dec, longer_res):
        for enc in single_enc.values():
            for name, res in enc.items():
                if "ohe" in name:
                    assert res.shape == (1, 2)
                else:
                    assert res.shape == (1,)
        for dec in single_dec.values():
            for res in dec.values():
                assert res.shape == (1,)
        n = self.longer.shape[0]
        for name, long in longer_res["enc"].items():
            if "ohe" in name:
                assert long.shape == (n, 2)
            else:
                assert long.shape == (n,)
        for long in longer_res["dec_np"].values():
            assert long.shape == (n,)
        for long in longer_res["dec_pd"].values():
            assert long.shape == (n,)

    def test_binary_decoding_name(self, single_dec):
        assert single_dec["pd"]["pos_res"].name == self.np_input.name
        assert single_dec["pd"]["neg_res"].name == self.pd_input.name
        assert single_dec["pd"]["pos_res_ohe"].name == "test"
        assert single_dec["pd"]["neg_res_ohe"].name == "test"

    def test_binary_decoding(self, single_dec, longer_res):
        for dec in single_dec.values():
            for name, res in dec.items():
                if "pos" in name:
                    assert np.all(res == self.pos)
                    assert np.all(res == self.pos_pd)
                if "neg" in name:
                    assert np.all(res == self.neg)
                    assert np.all(res == self.neg_pd)

        for long in longer_res["dec_np"].values():
            assert np.all(long == self.longer)
            assert np.all(long == self.longer_pd)
        for long in longer_res["dec_pd"].values():
            assert np.all(long == self.longer)
            assert np.all(long == self.longer_pd)

    def test_binary_encoding_width(self):
        assert self.np_input.encoding_width(True) == 2
        assert self.np_input.encoding_width(False) == 1
        assert self.pd_input.encoding_width(True) == 2
        assert self.pd_input.encoding_width(False) == 1

    def test_binary_fail_incorrect_vals(self):
        with pytest.raises(ValueError):
            Binary(np.array(["pos", "neg"]), ["positive", "negative"], "test")

    def test_binary_fail_many_vals(self):
        with pytest.raises(ValueError):
            Binary(np.array(["pos", "neg", "third_value"]), name="test")

    def test_binary_fail_no_name(self):
        with pytest.raises(ValueError):
            Binary(np.array(["pos", "neg"]))

    def test_binary_fail_empty(self):
        with pytest.raises(ValueError):
            Binary(np.array([]), name="test")

    def test_binary_fail_encode_invalid(self):
        with pytest.raises(ValueError):
            self.np_input.encode(np.array(["third value"]))
        with pytest.raises(ValueError):
            self.pd_input.encode(np.array(["third value"]))
        with pytest.raises(ValueError):
            self.np_input.encode(pd.Series(["third value"]))
        with pytest.raises(ValueError):
            self.pd_input.encode(pd.Series(["third value"]))

    def test_binary_fail_decode_invalid(self):
        with pytest.raises(ValueError):
            self.np_input.decode(np.array([2]))
        with pytest.raises(ValueError):
            self.pd_input.encode(np.array([[0, 1], [0, 9]]))
