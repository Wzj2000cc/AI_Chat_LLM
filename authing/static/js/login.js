function doLogin(){
    var account=$("#account").val();
    var pwd=$("#pwd").val();
    if(account==''){
        layer.msg('账号不能为空');
        return;
    }
     if(pwd==''){
        layer.msg('密码不能为空');
        return;
    }
     $.ajax({
        //请求方式
        type : "POST",
        //请求地址
        url : "./doLogin",
        //数据，json字符串
        data : {account:account,pwd:pwd},
        //请求成功
        success : function(resstr) {
            res=JSON.parse(resstr)
            if(res.code==200){
                location.href="./admin/home"
                layer.msg('登录成功');
            }else{
                layer.msg('账号或密码错误');
            }
        },
        //请求失败，包含具体的错误信息
        error : function(e){
            layer.msg('系统错误，稍后请重试');
        }
    });
}
// $.ajax({
//            //请求方式
//            type : "POST",
//            //请求的媒体类型
//            contentType: "application/json;charset=UTF-8",
//            //请求地址
//            url : "http://127.0.0.1/admin/list/",
//            //数据，json字符串
//            data : JSON.stringify(list),
//            //请求成功
//            success : function(result) {
//                console.log(result);
//            },
//            //请求失败，包含具体的错误信息
//            error : function(e){
//                console.log(e.status);
//                console.log(e.responseText);
//            }
//        });