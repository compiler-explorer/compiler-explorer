define(function (require) {
    "use strict";
    var _ = require('underscore');
    var $ = require('jquery');

    function get(domRoot) {
        var result = {};
        _.each(domRoot.find(".btn.active"), function (a) {
            result[$(a).data().bind] = true;
        });
        return result;
    }

    function Toggles(root, state) {
        this.domRoot = root;
        this.changeHandlers = [];
        state = state || this.get();
        this.domRoot.find('.btn')
            .click(_.bind(this.onClick, this))
            .each(function () {
                $(this).toggleClass('active', !!state[$(this).data().bind]);
            });
        this.state = get(this.domRoot);
    }

    Toggles.prototype.get = function () {
        return this.state;
    };

    Toggles.prototype.onClick = function (event) {
        var button = $(event.currentTarget);
        button.toggleClass('active');
        var before = this.state;
        var after = this.state = get(this.domRoot);
        _.each(this.changeHandlers, function (f) {
            f(before, after);
        });
    };

    Toggles.prototype.on = function (event, handler, that) {
        if (event === "change") {
            this.changeHandlers.push(_.bind(handler, that));
        }
    };

    return Toggles;
});
