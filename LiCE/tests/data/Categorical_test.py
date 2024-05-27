import numpy as np
import pandas as pd
import pytest

from LiCE.data.Features import Categorical

training_vector = ["a", "b", "c", "d", "a", "d", "a", "c"]
cat_values = ["a", "b", "c", "d"]
value_map = [0, 1, 2, 3]


class TestCategorical:
    np_input = Categorical(
        np.array(training_vector),
        cat_values,
        value_map,
        name="test",
    )
    pd_input = Categorical(
        pd.Series(training_vector, name="test"),
        cat_values,
        value_map,
    )

    dummy = np.array(cat_values)
    dummy_pd = pd.Series(cat_values)

    @pytest.fixture()
    def longer(self):
        return np.random.choice(cat_values, 100)

    @pytest.fixture()
    def longer_pd(self, longer):
        return pd.Series(longer)

    @pytest.fixture()
    def dummy_enc(self):
        return {
            "np": {
                "res": self.np_input.encode(self.dummy, normalize=True, one_hot=False),
                "res_ohe": self.np_input.encode(
                    self.dummy_pd, normalize=True, one_hot=True
                ),
            },
            "pd": {
                "res": self.np_input.encode(
                    self.dummy_pd, normalize=True, one_hot=False
                ),
                "res_ohe": self.np_input.encode(
                    self.dummy, normalize=True, one_hot=True
                ),
            },
        }

    @pytest.fixture()
    def dummy_dec(self, dummy_enc):
        return {
            "np": {
                "res": self.np_input.decode(
                    dummy_enc["np"]["res"], denormalize=True, return_series=False
                ),
                "res_ohe": self.pd_input.decode(
                    dummy_enc["np"]["res_ohe"], denormalize=True, return_series=False
                ),
            },
            "pd": {
                "res": self.pd_input.decode(
                    dummy_enc["pd"]["res"], denormalize=True, return_series=True
                ),
                "res_ohe": self.np_input.decode(
                    dummy_enc["pd"]["res_ohe"], denormalize=True, return_series=True
                ),
            },
        }

    @pytest.fixture()
    def longer_res(self, longer, longer_pd):
        enc = {
            "np": self.np_input.encode(longer, normalize=True, one_hot=False),
            "np_ohe": self.np_input.encode(longer, normalize=True, one_hot=True),
            "pd": self.pd_input.encode(longer_pd, normalize=True, one_hot=False),
            "pd_ohe": self.pd_input.encode(longer_pd, normalize=True, one_hot=True),
        }
        dec_np = {
            "np": self.pd_input.decode(
                enc["np"], denormalize=True, return_series=False
            ),
            "np_ohe": self.np_input.decode(
                enc["np_ohe"], denormalize=True, return_series=False
            ),
            "pd": self.np_input.decode(
                enc["pd"], denormalize=True, return_series=False
            ),
            "pd_ohe": self.pd_input.decode(
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

    def test_categorical_types(self, dummy_enc, dummy_dec):
        for enc in dummy_enc.values():
            for res in enc.values():
                assert isinstance(res, np.ndarray)
                assert res.dtype == np.float64
        for res in dummy_dec["np"].values():
            assert isinstance(res, np.ndarray)
        for res in dummy_dec["pd"].values():
            assert isinstance(res, pd.Series)

    def test_categorical_is_correct(self, dummy_enc, longer_res):
        for enc in dummy_enc.values():
            assert np.all(enc["res"] == np.arange(len(cat_values)))
            assert np.all(enc["res_ohe"] == np.eye(len(cat_values)))
        for key in dummy_enc["np"].keys():
            assert np.all(dummy_enc["np"][key] == dummy_enc["pd"][key])

        for name, res in longer_res["enc"].items():
            if "ohe" in name:
                assert np.all(res.sum(axis=1) == 1)
                assert np.all(np.isin(res, [0, 1]))

    def test_categorical_keeps_shape(self, dummy_enc, dummy_dec, longer, longer_res):
        for enc in dummy_enc.values():
            assert enc["res"].shape == (self.dummy.shape[0],)
            assert enc["res_ohe"].shape == (self.dummy.shape[0], len(cat_values))
        for dec in dummy_dec.values():
            for res in dec.values():
                assert res.shape == self.dummy.shape

        n = longer.shape[0]
        for name, long in longer_res["enc"].items():
            if "ohe" in name:
                assert long.shape == (n, len(cat_values))
            else:
                assert long.shape == (n,)
        for long in longer_res["dec_np"].values():
            assert long.shape == (n,)
        for long in longer_res["dec_pd"].values():
            assert long.shape == (n,)

    def test_categorical_decoding_name(self, dummy_dec):
        assert self.np_input.name == "test"
        assert self.pd_input.name == "test"
        assert dummy_dec["pd"]["res_ohe"].name == self.np_input.name
        assert dummy_dec["pd"]["res"].name == self.pd_input.name
        assert dummy_dec["pd"]["res"].name == "test"
        assert dummy_dec["pd"]["res_ohe"].name == "test"

    def test_categorical_decoding(self, dummy_dec, longer, longer_pd, longer_res):
        for dec in dummy_dec.values():
            for res in dec.values():
                assert np.all(res == self.dummy)
                assert np.all(res == self.dummy_pd)

        for long in longer_res["dec_np"].values():
            assert np.all(long == longer)
            assert np.all(long == longer_pd)
        for long in longer_res["dec_pd"].values():
            assert np.all(long == longer)
            assert np.all(long == longer_pd)

    def test_categorical_encoding_width(self):
        assert self.np_input.encoding_width(True) == len(cat_values)
        assert self.np_input.encoding_width(False) == 1
        assert self.pd_input.encoding_width(True) == len(cat_values)
        assert self.pd_input.encoding_width(False) == 1

    def test_categorical_value_mapping(self):
        assert self.np_input.value_mapping == {
            val: i for i, val in zip(value_map, cat_values)
        }

    def test_categorical_defaults(self):
        unspecified = Categorical(np.array(cat_values), name="test")
        assert self.np_input.value_mapping == unspecified.value_mapping

    def test_categorical_fail_incorrect_vals(self):
        with pytest.raises(ValueError):
            Categorical(np.array(["a", "b"]), ["A", "B"], "test")

    def test_categorical_fail_empty(self):
        with pytest.raises(ValueError):
            Categorical(np.array([]), name="test")

    def test_categorical_fail_no_name(self):
        with pytest.raises(ValueError):
            Categorical(np.array(["a"]))

    def test_categorical_fail_encode_invalid(self):
        with pytest.raises(ValueError):
            self.np_input.encode(np.array(["third value"]))
        with pytest.raises(ValueError):
            self.pd_input.encode(np.array(["third value"]))
        with pytest.raises(ValueError):
            self.pd_input.encode(pd.Series(["third value"]))

    def test_categorical_fail_decode_invalid(self):
        # wrong value
        with pytest.raises(ValueError):
            self.np_input.decode(np.array([-1]))
        # wrong one hot value
        with pytest.raises(ValueError):
            self.pd_input.encode(np.array([[0, 1, 0, 0], [0, 0, 0, 2]]))
        # wrong shape
        with pytest.raises(ValueError):
            self.pd_input.encode(np.array([[0, 1, 0], [0, 0, 1]]))
